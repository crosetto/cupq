/*---------------------------------------------------------------------

Copyright 2019 Paolo G. Crosetto <pacrosetto@gmail.com>
SPDX-License-Identifier: Apache-2.0

---------------------------------------------------------------------*/
#pragma once
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>

#include "node.h"

#ifdef __CUDACC__
#define H_F __host__
#define D_F __device__
#define H_D_F __host__ __device__
#define ALIGN //__align__(8)
#else
#define H_F
#define D_F
#define H_D_F
#define ALIGN //__align__(8)
#endif

namespace cupq {

/**
   Implementation of a warp-level heap data structure in CUDA
*/
template <typename TValue, typename TIndex, int WarpSize, int Arity,
          int warpsPerBlock, int Limit>
struct ALIGN Heap {
  const static int arity = Arity;
  using node_t = Node<TValue, TIndex, WarpSize>;
  using value_t = TValue;
  using index_t = TIndex;

  H_D_F
  Heap() {}

  /** sets the position of the last valid element whithin the last warp */
  D_F void setPosition(int &sPosition, int pos) {
    if (threadIdx.x == 0)
      sPosition = pos;
    __syncwarp();
  }

  /** constructs the data structure given the addresses to the shared memory
     quantities, i.e. the first elements of each warp, the position of the last
     valid element in the last warp, and the root element of the heap*/
  D_F Heap(Array<value_t, Limit> &tops_, int &position_, index_t &top2_)
      : mSize(1) {
    setPosition(position_, 0);
    for (int i = 0; i < Limit; ++i) {
      mData[i].init();
      // setTop(tops_, i, 0x0fffffff);
    }

    for (int i = 0; i < Limit; i += 32) {
      if (threadIdx.x + i < Limit)
        tops_[threadIdx.x + i] = 0x0fffffff;
    }
    __syncwarp();
    updateTop2(top2_);
  }

  D_F Heap(Heap const &other_) : mSize(other_.mSize), mData(other_.mData) {}

  /** updates the value at the root node,
      copying in ut the smallest element of the root warp*/
  D_F void updateTop2(index_t &top2_) { setTop2(top2_, mData[0].mData2); }

  /** sets the value at the root node to a value given in input*/
  D_F void setTop2(index_t &sTop2, index_t top2_) {
    if (threadIdx.x == 0)
      sTop2 = top2_;
    __syncwarp();
  }

  /**Initializes the shared memory*/
  D_F void setup(Array<value_t, Limit> &tops_, int &position_, index_t &top2_) {
    mSize = 1;

    setPosition(position_, 0);
    for (int i = 0; i < mData.size(); ++i) {
      mData[i].init();
    }

    for (int i = 0; i < Limit; i += 32) {
      if (threadIdx.x + i < Limit)
        tops_[threadIdx.x + i] = 0x0fffffff;
    }
    __syncwarp();
    updateTop2(top2_);
  }

  /**computes the index of the parent warp, given the index of a child*/
  D_F int parent(int index) {
    if (arity == 2)
      return (index - index % arity) / arity;
    else {
      if (index == 1)
        return 1;
      else {
        return ((index - 2) - (index - 2) % arity) / arity + 1;
      }
    }
  }

  /**computes the index of a child warp, given the index of the parent and an
     ordinal nuber
     @param index the index of the parent
     @param id an integer identifying one child among the children (the id-th)
   */
  D_F int child(int index, int id) { return 1 + (index)*arity + id; }

  template <typename T> D_F T min(T a, T b) { return a < b ? a : b; }

  template <typename P, typename T> D_F T min(P predicate, T &&a, T &&b) {
    return predicate(a, b) ? std::forward<T>(a) : std::forward<T>(b);
  }

  /** returning an index identifying one child (among all the children of a
     node) having the smallest "top" value */
  D_F int minChild(Array<value_t, Limit> &sTops, int index) {
    static_assert(Arity == 32, "if you want to generalize change also the "
                               "lines below (assuming 32-way heap)");

    auto firstchild_ = child(index, 0);
    bool leaf = firstchild_ > mSize; // index is a leaf, no childs
    int nchild = (mSize - 1) % Arity;

    if (leaf && nchild == 1) {
      return firstchild_;
    }

    index_t ret, ret1;
    value_t reg0, reg1;

    ret = threadIdx.x;

    int id = child(index, threadIdx.x);
    if (id >= mSize)
      reg0 =
          0x0fffffff; // avoid access out of bound (and huge shmem allocation)
    else
      reg0 = sTops[id]; // mData[id].top();

    reg1 = __shfl_down_sync(0xffffffff, reg0, 1);

    for (int i = 1; i <= 16; i *= 2) {
      ret1 = __shfl_down_sync(0xffffffff, ret, i);
      if (reg0 > reg1) {
        ret = ret1;
      }
      if (leaf && nchild <= i) {
        ret = __shfl_sync(0xffffffff, ret, 0, WarpSize);
        return child(index, ret);
      }
      if (reg0 > reg1) {
        reg0 = reg1;
      }
      reg1 = __shfl_down_sync(0xffffffff, reg0, i * 2);
    }

    ret = __shfl_sync(0xffffffff, ret, 0);
    return child(index, ret);
  }

  /** inserting a node to the end of the heap*/
  D_F void insert(DevicePair<value_t, index_t> n, Array<value_t, Limit> &sTops,
                  int &sPosition, index_t &sTop2) {

    if (mSize == 0) {
      mData[mSize].init(n);
      setTop(sTops, mSize, n.first);
      ++mSize;
      setPosition(sPosition, 1);
    } else {
      if (n.first > sTops[mSize - 1])
        if (sPosition != WarpSize) {
          mData[mSize - 1].replaceDiscard(n); // throw away result
          updateTop(sTops, mSize - 1);
          int pos = sPosition + 1;
          setPosition(sPosition, pos);
        } else { // grow graph
          mData[mSize].init(n);
          setTop(sTops, mSize, n.first);
          ++mSize;
          setPosition(sPosition, 1);
        }
      else {
        if (sPosition != WarpSize) {
          if (mSize > 1) {
            mData[mSize - 1].replaceDiscard(
                propagateUp(n, mSize, sTops)); // throw away return value
            updateTop(sTops, mSize - 1);
            int pos = sPosition + 1;
            setPosition(sPosition, pos);
          } else {
            mData[0].replaceDiscard(n); // throw away return value
            updateTop(sTops, 0);
            int pos = sPosition + 1;
            setPosition(sPosition, pos);
          }
        } else {
          auto &&tmp = propagateUp(n, mSize, sTops);
          mData[mSize].init(tmp); // grow graph
          setTop(sTops, mSize, tmp.first);
          ++mSize;
          setPosition(sPosition, 1);
        }
      }
    }
    updateTop2(sTop2);
  }

  D_F void updateTop(Array<value_t, Limit> &sTops, unsigned id) {
    setTop(sTops, id, mData[id].mData);
  }

  D_F void setTop(Array<value_t, Limit> &sTops, unsigned id, value_t data) {
    if (threadIdx.x == 0)
      sTops[id] = data;
    __syncwarp();
  }

  D_F index_t pop(Array<value_t, Limit> &sTops, int &sPosition,
                  index_t &sTop2) {

    index_t ret = sTop2;
    if (sPosition == 0) {
      mData[mSize - 1].mData = 0x0fffffff; // reset all
      mData[mSize - 1].mData2 = 0;         // reset all
      updateTop(sTops, mSize - 1);
      --mSize;
      setPosition(sPosition, WarpSize);
    }

    if (mSize > 0) { // otherwise do nothing (popping from an empty container)
      if (sPosition == 0)
        setTop(sTops, 0, 0x0fffffff);
      mData[0].replaceSmallestDiscard(mData[mSize - 1].pop(sPosition));
      int pos = sPosition - 1;
      setPosition(sPosition, pos);
      updateTop(sTops, mSize - 1);

      updateTop(sTops, 0);
      if (mSize > 1) {
        auto sTop2 = __shfl_sync(0xffffffff, mData[0].mData2, 0, WarpSize);
        DevicePair<value_t, index_t> tmp =
            propagateDown(make_DevicePair(sTops[0], sTop2), sTops);
        if (threadIdx.x == 0) {
          mData[0].mData = tmp.first;
          mData[0].mData2 = tmp.second;
        }
        updateTop(sTops, 0);
      }
    }
    updateTop2(sTop2);
    return ret;
  }

  // todo index in "parent" function starts from 1 instead of 0. crap
  D_F __forceinline__ DevicePair<value_t, index_t>
  propagateUp(DevicePair<value_t, index_t> n, int index,
              Array<value_t, Limit> &sTops) {

    Array<int, LOGLIMIT> stackChild; // "no worries", this fits in L1 cache
    int pos = 0;
    while (index > 1) {
      if (n.first >= sTops[index - 1]) {
        break;
      } else {
        stackChild[pos] = (index - 2) % Arity;
        index = parent(index);
        ++pos;
      }
    }
    DevicePair<value_t, index_t> ret = mData[index - 1].replace(n);
    updateTop(sTops, index - 1);
    // unwind stack
    while (pos) {
      --pos;

      index = child(index - 1, stackChild[pos]);

      ret = mData[index].replace(ret);
      if (pos)
        updateTop(sTops, index);
    }
    return ret;
  }

  D_F __forceinline__ DevicePair<value_t, index_t>
  propagateDown(DevicePair<value_t, index_t> n, Array<value_t, Limit> &sTops) {

    if (mSize < 2) // no children
      return n;

    auto mchild = minChild(sTops, 0);

    DevicePair<value_t, index_t> ret = n;

    if (n.first < sTops[mchild])
      // heap property is OK, just return
      return n;

    // set the return value and continue below
    ret = mData[mchild].replaceSmallest(n);
    updateTop(sTops, mchild);

    if (mSize <= Arity + 1) // index is a leaf already (1<=index<=33)
      return ret;

    auto index = mchild;

    index_t sTop2 = mData[index].mData2;

    // broadcast
    sTop2 = __shfl_sync(0xffffffff, sTop2, 0, WarpSize);

    bool leaf = child(index, 0) >= mSize;
    while (!leaf) {
      mchild = minChild(sTops, index);

      if (sTops[index] <= sTops[mchild]) {
        // OK, heap property is fine
        break;
      } else {

        n = make_DevicePair(sTops[index], sTop2);
        // swap the min in the two warps (parent and child):
        // do the swap between the 2 warps (a double replace)
        if (threadIdx.x == 0) {
          mData[index].mData = mData[mchild].mData;
          mData[index].mData2 = mData[mchild].mData2;
        }

        mData[mchild].replaceSmallestDiscard(
            n); // throw away the returned value (used it already above)
        updateTop(sTops, mchild);
        updateTop(sTops, index);

        // get the next min
        sTop2 = mData[mchild].mData2;
        sTop2 = __shfl_sync(0xffffffff, sTop2, 0, WarpSize);
      }

      index = mchild;
      leaf = child(index, 0) >= mSize;
    }

    return ret;
  }

  D_F int size() const { return mSize; }

  D_F void print() {
    int exp = 1;
    for (int i = 1; i <= mSize; ++i) {
      if (i == exp) {
        if (threadIdx.x == 0 && blockIdx.x == 0)
          printf("\n");
        exp *= arity;
      }
      __syncthreads();
      mData[i - 1].print();
    }
    if (threadIdx.x == 0 && blockIdx.x == 0)
      printf("\n =============== \n");
  }

private:
  Array<node_t, Limit> mData; /* local memory */
  int mSize = 0;              /* local memory (cached) */
};
} // namespace cupq
#undef H_F
#undef D_F
#undef H_D_F
#undef ALIGN
