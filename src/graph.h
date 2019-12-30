/*---------------------------------------------------------------------

Copyright 2019 Paolo G. Crosetto <pacrosetto@gmail.com>
SPDX-License-Identifier: Apache-2.0

---------------------------------------------------------------------*/
#pragma once
#include <fstream>
#include <iostream>
#include <regex>
#include <vector>

#ifdef __CUDACC__
#define H_D_F __host__ __device__
#define H_F __host__
#else
#define H_D_F
#define H_F
#endif

namespace cupq {

/** Graph data read from input file
 */
template <typename TTValue, typename TTIndex, Backend TBackend> class Graph {
public:
  static const Backend backend = TBackend;
  using value_t = TTValue;
  using index_t = TTIndex;

  Graph() : m_size(0), m_source_offsets(), m_weights(), m_destinations() {}

  Graph(Graph const &other_)
      : m_size(other_.m_size), m_source_offsets(other_.m_source_offsets),
        m_weights(other_.m_weights), m_destinations(other_.m_destinations) {}

  Graph(Vector<index_t, TBackend> const &source_offsets_,
        Vector<value_t, TBackend> const &weights_,
        Vector<index_t, TBackend> const &destinations_)
      : m_size(source_offsets_.size()), m_source_offsets(source_offsets_),
        m_weights(weights_), m_destinations(destinations_) {}

  Graph(Vector<index_t, TBackend> &&source_offsets_,
        Vector<value_t, TBackend> const &weights_,
        Vector<index_t, TBackend> const &destinations_)
      : m_size(source_offsets_.size()), m_source_offsets(source_offsets_),
        m_weights(weights_), m_destinations(destinations_) {}

  // does a deep copy of the graph, export to CPU or GPU
  template <Backend b> Graph<TTValue, TTIndex, b> clone() {
    Graph<TTValue, TTIndex, b> ret;
    ret.m_source_offsets.resize(m_source_offsets.size());
    for (unsigned i = 0; i < m_source_offsets.size(); ++i)
      ret.m_source_offsets[i] = m_source_offsets[i];
    ret.m_weights.resize(m_weights.size());
    for (unsigned i = 0; i < m_weights.size(); ++i)
      ret.m_weights[i] = m_weights[i];
    ret.m_destinations.resize(m_destinations.size());
    for (unsigned i = 0; i < m_destinations.size(); ++i)
      ret.m_destinations[i] = m_destinations[i];
    ret.m_size = m_size;
    return ret;
  }

  // reads a graph from the DIMACS database ("forward" mode)
  H_F int readDIMACS(char const *filename) {
    std::ifstream fin_(filename, std::ios_base::in);
    if (!fin_) {
      std::cout << "error reading file\n";
      return -1;
    }

    std::string line_;
    std::regex re1(std::string("p sp (.*) (.*)"));
    std::cmatch m1;
    while (std::getline(fin_, line_) &&
           !std::regex_search(line_.c_str(), m1, re1)) {
    }

    index_t nnodes_ = atoi(m1[1].str().c_str());
    index_t nedges_ = atoi(m1[2].str().c_str());

    std::vector<index_t> origins_;
    std::vector<index_t> destinations_;
    std::vector<value_t> weights_;
    std::vector<index_t> permutation_(nedges_);
    index_t k = 0;
    for (auto &i : permutation_) {
      i = k++;
    }
    origins_.reserve(nedges_);
    destinations_.reserve(nedges_);
    weights_.reserve(nedges_);

    while (std::getline(fin_, line_)) {
      std::regex re2(std::string("a (.*) (.*) (.*)"));
      std::cmatch m2;
      if (std::regex_search(line_.c_str(), m2, re2)) {
        origins_.push_back(atoi(m2[1].str().c_str()) -
                           1); // numeration starts from 1 in the file
        destinations_.push_back(atoi(m2[2].str().c_str()) - 1);
        weights_.push_back(atof(m2[3].str().c_str()));
      }
    }

    std::sort(permutation_.begin(), permutation_.end(),
              [&origins_](auto const &i, auto const &j) {
                return origins_[i] < origins_[j];
              });

    std::vector<index_t> new_destinations_(nedges_);
    std::vector<index_t> new_origins_(nedges_);
    std::vector<value_t> new_weights_(nedges_);

    for (index_t i = 0; i < nedges_; ++i) {
      new_destinations_[i] = destinations_[permutation_[i]];
      new_origins_[i] = origins_[permutation_[i]];
      new_weights_[i] = weights_[permutation_[i]];
    }

    m_source_offsets.reserve(nnodes_ + 1);
    m_destinations.reserve(nedges_);
    m_weights.reserve(nedges_);

    setOnHost();

    m_source_offsets.push_back((index_t)0);

    index_t count_edges_ = 0;
    index_t prev_count_ = 0;
    index_t new_orig_ = 0;
    for (index_t i = 0; i < nedges_; ++i) {
      index_t orig_ = new_orig_;
      new_orig_ = new_origins_[i];
      if (new_orig_ > orig_) {
        for (int l = 0; l < new_orig_ - orig_ - 1;
             ++l) { // fill the gap repeating the last number
          m_source_offsets.push_back(prev_count_);
        }
        m_source_offsets.push_back(count_edges_);
        prev_count_ = count_edges_;
      }
      count_edges_++;

      m_destinations.push_back(new_destinations_[i]);
      m_weights.push_back(new_weights_[i]);
    }

    m_size = nnodes_; // maximum of the nodes
    m_source_offsets.push_back(count_edges_);
    return 0;
  }

  // reads a graph from the DIMACS database
  std::vector<std::pair<value_t, value_t>>
  readDIMACSCoordinates(char const *filename) {
    std::ifstream fin_(filename, std::ios_base::in);
    if (!fin_) {
      std::cout << "error reading file\n";
      return {};
    }

    std::string line_;
    std::regex re1(std::string("p aux sp co (.*)"));
    std::cmatch m1;
    while (std::getline(fin_, line_) &&
           !std::regex_search(line_.c_str(), m1, re1)) {
    }
    std::regex_search(line_.c_str(), m1, re1);

    index_t nnodes_ = atoi(m1[1].str().c_str());

    std::vector<std::pair<value_t, value_t>> coordinates;
    coordinates.resize(0);
    coordinates.reserve(nnodes_);

    while (std::getline(fin_, line_)) {
      std::regex re2(std::string("v [0-9].* (.*) (.*)"));
      std::cmatch m2;
      if (std::regex_search(line_.c_str(), m2, re2)) {
        coordinates.push_back(
            std::make_pair((value_t)(atof(m2[1].str().c_str())),
                           (value_t)(atof(m2[2].str().c_str()))));
      }
    }

    return coordinates;
  }

  Graph reverse() const {
    Graph ret;
    ret.m_source_offsets.reserve(m_source_offsets.size());
    ret.m_destinations.reserve(m_destinations.size());
    ret.m_weights.reserve(m_weights.size());

    std::vector<unsigned> adj_source;
    adj_source.reserve(m_destinations.size());
    std::vector<std::pair<value_t, index_t>> adj_destinations(
        m_destinations.size(), std::make_pair(0.f, -1));

    unsigned Id = 0;
    unsigned l = 0;

    for (auto i = 0; i < m_source_offsets.size() - 1; ++i) {
      for (auto j = m_source_offsets[i]; j < m_source_offsets[i + 1]; ++j) {
        Id = i;
        adj_source.push_back(Id);
        adj_destinations[l] = std::make_pair(m_weights[l], m_destinations[l]);
        ++l;
      }
    }
    for (auto j = m_source_offsets[m_source_offsets.size() - 1];
         j < adj_destinations.size(); ++j) {
      adj_source.push_back(m_source_offsets.size() - 1);
      adj_destinations[l] = std::make_pair(m_weights[l], m_destinations[l]);
      ++l;
    }

    if (l != adj_destinations.size())
      std::cout << "ERROR in transpose graph, " << l << " should be "
                << adj_destinations.size() << "\n";

    std::vector<index_t> reindex(m_destinations.size());
    unsigned k = 0;
    for (auto &i : reindex)
      i = k++;
    //"stabilized" sort
    std::sort(reindex.begin(), reindex.end(),
              [&adj_destinations, &adj_source](auto const &i, auto const &j) {
                return adj_destinations[i].second +
                           ((float)adj_source[i] + 1) / adj_source.size() <
                       adj_destinations[j].second +
                           ((float)adj_source[j] + 1) / adj_source.size();
              });

    std::vector<index_t> adj_source_new(m_destinations.size(), 0);
    std::vector<std::pair<value_t, index_t>> adj_destinations_new(
        m_destinations.size(), std::make_pair(0.f, -1));

    k = 0;
    for (auto &i : adj_destinations_new) {
      i = adj_destinations[reindex[k++]];
    }
    k = 0;
    for (auto &i : adj_source_new)
      i = adj_source[reindex[k++]];

    // recompress transposed
    index_t dest_ = 0;
    index_t new_dest_ = 0;
    ret.m_source_offsets.push_back(0);
    for (auto i = 0; i < adj_destinations_new.size(); ++i) {
      new_dest_ = adj_destinations_new[i].second;
      if (new_dest_ > dest_) {
        for (int l = 0; l < new_dest_ - dest_ - 1;
             ++l) // fill the gap repeating the last number
          ret.m_source_offsets.push_back(i);
        ret.m_source_offsets.push_back(i);
      }

      dest_ = new_dest_;
      ret.m_destinations.push_back((index_t)adj_source_new[i]);
      ret.m_weights.push_back((value_t)adj_destinations_new[i].first);
    }
    ret.m_source_offsets.push_back((int)adj_destinations_new.size());
    ret.m_size = m_size;

    return ret;
  }

  H_F void randomNoise() {
    for (auto i = 0; i < m_destinations.size(); ++i)
      m_weights[i] +=
          static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / .01f));
  }

  H_D_F
  void setup() {
    m_size = 0u;
    m_source_offsets.setup();
    m_weights.setup();
    m_destinations.setup();
  }

  H_D_F
  index_t const &sourceOffsets(unsigned id) const {
    return m_source_offsets[id];
  }
  H_D_F
  index_t const &destinationIndices(unsigned id) const {
    return m_destinations[id];
  }
  H_D_F
  TTValue const &weights(unsigned id) const { return m_weights[id]; }

  H_D_F
  index_t const &getSourceDegrees(unsigned id) const {
    return m_source_offsets[id];
  }
  H_D_F
  index_t const &getDestinations(unsigned id) const {
    return m_destinations[id];
  }
  H_D_F
  TTValue const &getWeights(unsigned id) const { return m_weights[id]; }

  H_D_F
  size_t size() const { return m_size; }

  H_D_F
  size_t nbNodes() const { return m_size; }

  H_D_F
  size_t nbEdges() const { return m_destinations.size(); }

  // unsafe getters for the nvgraph API is accepting non-const pointers
  H_D_F
  index_t &sourceOffsetsUnsafe(unsigned id) { return m_source_offsets[id]; }
  H_D_F
  index_t &destinationIndicesUnsafe(unsigned id) { return m_destinations[id]; }
  H_D_F
  TTValue &weightsUnsafe(unsigned id) { return m_weights[id]; }

  // unsafe getters for the nvgraph API is accepting non-const pointers
  H_D_F
  index_t &sourceDegrees(unsigned id) { return m_source_offsets[id]; }
  H_D_F
  index_t &destinations(unsigned id) { return m_destinations[id]; }
  H_D_F
  TTValue &weights(unsigned id) { return m_weights[id]; }

  H_F void free() {
    m_source_offsets.free();
    m_destinations.free();
    m_weights.free();
  }

  void print() {
    std::cout << "source offsets\n";
    for (auto &i : m_source_offsets)
      std::cout << i << " ";
    std::cout << "\n";

    std::cout << "weights\n";
    for (unsigned i = 0; i < m_destinations.size(); ++i)
      std::cout << m_weights[i] << " ";
    std::cout << "\n";

    std::cout << "destinations\n";
    for (unsigned i = 0; i < m_destinations.size(); ++i)
      std::cout << m_destinations[i] << " ";
    std::cout << "\n";
  }

  void copyToDevice() {
    m_source_offsets.copyToDevice();
    m_weights.copyToDevice();
    m_destinations.copyToDevice();
  }

  void copyToHost() {
    m_source_offsets.copyToHost();
    m_weights.copyToHost();
    m_destinations.copyToHost();
  }

  void setOnDevice() {
    m_source_offsets.setOnDevice();
    m_weights.setOnDevice();
    m_destinations.setOnDevice();
  }

  void setOnHost() {
    m_source_offsets.setOnHost();
    m_weights.setOnHost();
    m_destinations.setOnHost();
  }

private:
  size_t m_size;
  Vector<index_t, TBackend> m_source_offsets;
  Vector<value_t, TBackend> m_weights;
  Vector<index_t, TBackend> m_destinations;
};

template <typename Graph> struct is_graph : std::false_type {};

template <typename TTValue, typename TTIndex, Backend TBackend>
struct is_graph<Graph<TTValue, TTIndex, TBackend>> : std::true_type {};
} // namespace cupq
#undef H_D_F
#undef H_F
