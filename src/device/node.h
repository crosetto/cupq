/*---------------------------------------------------------------------

Copyright 2019 Paolo G. Crosetto <pacrosetto@gmail.com>
SPDX-License-Identifier: Apache-2.0

---------------------------------------------------------------------*/
#pragma once

#define H_F __host__
#define D_F __device__
#define H_D_F __host__ __device__
#define ALIGN // __align__(8)

namespace cupq {
template <typename Type, typename Type2, int WarpSize> struct ALIGN Node {

  static constexpr int warp_size = WarpSize;

  D_F Node(Node const &other_) : mData(other_.mData), mData2(other_.mData2){};

  D_F Node() {
    mData = 0x0fffffff;
    mData2 = 0;
  }

  D_F Node(DevicePair<Type, Type2> n) {
    if (threadIdx.x == 0) {
      mData = n.first;
      mData2 = n.second;
    } else {
      mData = 0x0fffffff;
      mData2 = 0;
    }
  }

  D_F void init() {
    mData = 0x0fffffff;
    mData2 = 0;
  }

  D_F void init(DevicePair<Type, Type2> const &n) {
    if (threadIdx.x == 0) {
      mData = n.first;
      mData2 = n.second;
    } else {
      mData = 0x0fffffff;
      mData2 = 0;
    }
  }

  D_F void replaceDiscard(DevicePair<Type, Type2> value) {
    // instructions in PTX:
    // create bitmask 00000111
    int vote = __ballot_sync(0xffffffff, mData < value.first);
    // find the position of first "1" in the warp
    int b;
    ++vote; // 00001000
    asm("bfind.u32 %0, %1;" : "=r"(b) : "r"(vote));
    // shift right some of the lanes (the "1"s)
    if (b == -1 /*I'm the largest*/)
      return;

    if (threadIdx.x >= b && threadIdx.x < WarpSize) {
      auto p = make_DevicePair(mData, mData2);
      mData = __shfl_up_sync(0xffffffff - ((1 << b) - 1), p.first, 1);
      mData2 = __shfl_up_sync(0xffffffff - ((1 << b) - 1), p.second, 1);
    }
    // insert in the empty spot
    if (threadIdx.x == b) {
      mData = value.first;
      mData2 = value.second;
    }
  }

  D_F DevicePair<Type, Type2> replace(DevicePair<Type, Type2> value) {
    // instructions in PTX:
    // pop the largest
    DevicePair<Type, Type2> ret(mData, mData2);
    // broadcast the return
    // __syncwarp();
    ret.first = __shfl_sync(0xffffffff, mData, WarpSize - 1, WarpSize);
    ret.second = __shfl_sync(0xffffffff, mData2, WarpSize - 1, WarpSize);
    // create bitmask 00000111
    int vote = __ballot_sync(0xffffffff, mData < value.first);
    // find the position of first "1" in the warp
    int b;
    ++vote;
    asm("bfind.u32 %0, %1;" : "=r"(b) : "r"(vote));
    // shift right some of the lanes (the "1"s)
    // __syncwarp();
    if (b == -1 /*I'm the largest*/)
      return value;

    if (threadIdx.x >= b && threadIdx.x < WarpSize) {
      mData = __shfl_up_sync(0xffffffff - ((1 << b) - 1), mData, 1);
      mData2 = __shfl_up_sync(0xffffffff - ((1 << b) - 1), mData2, 1);
    }
    // insert in the empty spot
    if (threadIdx.x == b) {
      mData = value.first;
      mData2 = value.second;
    }
    return ret;
  }

  D_F DevicePair<Type, Type2> replaceSmallest(DevicePair<Type, Type2> value) {
    // instructions in PTX:
    // pop the smallest
    DevicePair<Type, Type2> ret(mData, mData2);
    // broadcast the return
    ret.first = __shfl_sync(0xffffffff, mData, 0, WarpSize);
    ret.second = __shfl_sync(0xffffffff, mData2, 0, WarpSize);
    // create bitmask 00000111
    int vote = __ballot_sync(0xffffffff, mData <= value.first);
    // find the position of first "1" in the warp
    int b;
    asm("bfind.u32 %0, %1;" : "=r"(b) : "r"(vote));
    if (b == -1 /*I'm the smallest*/) {
      return value;
    }
    // shift left some of the lanes
    if (threadIdx.x <= b) {
      mData = __shfl_down_sync(vote, mData, 1);
      mData2 = __shfl_down_sync(vote, mData2, 1);
    }
    // insert in the empty spot
    if (threadIdx.x == b) {
      mData = value.first;
      mData2 = value.second;
    }
    return ret;
  }

  D_F void replaceSmallestDiscard(DevicePair<Type, Type2> value) {
    // instructions in PTX:
    // create bitmask 00000111
    int vote = __ballot_sync(0xffffffff, mData <= value.first);
    // find the position of first "1" in the warp
    int b;
    asm("bfind.u32 %0, %1;" : "=r"(b) : "r"(vote));
    if (b == -1 /*I'm the smallest*/) {
      return;
    }
    if (threadIdx.x <= b) {
      mData = __shfl_down_sync(vote, mData, 1);
      mData2 = __shfl_down_sync(vote, mData2, 1);
    }
    // insert in the empty spot
    if (threadIdx.x == b) {
      mData = value.first;
      mData2 = value.second;
    }
  }

  D_F DevicePair<Type, Type2> pop(int &position_) {
    DevicePair<Type, Type2> ret{(Type)0x0fffffff, 0};

    ret.first = __shfl_sync(0xffffffff, mData, position_ - 1, WarpSize);
    ret.second = __shfl_sync(0xffffffff, mData2, position_ - 1, WarpSize);
    if (threadIdx.x == position_ - 1) {
      mData = 0x0fffffff;
      mData2 = 0;
    }

    return ret;
  }

  D_F void print() {
    printf("%f:%d ", mData, mData2);
    __syncthreads();
    if (threadIdx.x == 0) {
      printf("            ");
    }
  }

  Type mData;   /* register */
  Type2 mData2; /* register */
};
} // namespace cupq
#undef H_F
#undef D_F
#undef H_D_F
#undef ALIGN
