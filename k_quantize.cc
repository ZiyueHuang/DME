#include <fstream>
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <random>
#include <unordered_map>


double log2_factorial(int n) {
  double sum = 0;
  for (int i = 1; i <= n; i++) sum += log2(i);
  return sum;
}

double log2_comb(int n, int k) {
  return log2_factorial(n) - log2_factorial(n - k) - log2_factorial(k);
}



void KQuantize(const std::vector< std::vector<double> >& vecs, const int k, const size_t num_nodes,
                float *sum, float *approx_sum, const size_t dim) {
  for (size_t i = 0; i < dim; i++) {
    sum[i] = 0;
    approx_sum[i] = 0;
  }

  double s[num_nodes];
  double x_min[num_nodes];

  for (size_t i = 0; i < num_nodes; i++) {
    const auto& data = vecs[i];
    s[i] = 0;
    x_min[i] = data[0];
    for (size_t j = 0; j < dim; j++) {
      s[i] += data[j] * data[j];
      if (x_min[i] > data[j]) x_min[i] = data[j];
    }
    s[i] = std::sqrt(2*s[i]);
  }

  static std::default_random_engine rng;
  std::uniform_real_distribution<double> unif(0, 1);

  double comm_kq = 0;

  for (size_t i = 0; i < num_nodes; i++) {
    const auto& data = vecs[i];
    std::unordered_map<int, int> table;
    for (size_t j = 0; j < dim; j++) {
      int level = 0;
      while ((x_min[i] + level * s[i] / (k - 1)) <= data[j]) level++;
      level--;
      if (level < 0 || level >= k) {
        assert(false);
        if (level < 0) level = 0;
        if (level >= k) level = k-1;
      }
      double step = s[i] / (k - 1);
      double lvalue = x_min[i] + level * step;
      double rvalue = x_min[i] + (level+1) * step;
      double value = (unif(rng) < (data[j] - lvalue) / step) ? rvalue : lvalue;

      sum[j] += data[j];
      approx_sum[j] += value;

      int l = value == lvalue ? level : (level+1);

      if (table.count(l)) {
        table[l]++;
      } else {
        table[l] = 1;
      }
    }

    double comm = 0;
    for (auto item : table) {
      comm += item.second/static_cast<double>(dim) * log2(static_cast<double>(dim)/item.second);
    }
    comm *= dim;
    comm += 2 + 32 + log2_comb(dim + k - 1, k - 1);
    comm_kq += comm;
  }

  double err = 0;
  double l2_norm = 0;
  for (size_t i = 0; i < dim; i++) {
    err += (sum[i] - approx_sum[i]) * (sum[i] - approx_sum[i]);
    l2_norm += sum[i] * sum[i];
  }

  std::cout << "comm_KQ " << comm_kq/num_nodes << std::endl;
  std::cout << "avg_error " << err / dim << std::endl;
}


int main() {
  const size_t num_nodes = 16;
  const size_t dim = 10000;
  std::vector< std::vector<double> > vecs;
  vecs.resize(num_nodes);
  std::vector<double> l1_vec;
  double l1_total = 0;
  const double factor = 1;

  for (size_t i = 0; i < num_nodes; i++) {
    char buffer[100];
    sprintf (buffer, "./data/data_%d.txt", i);
    std::ifstream infile(buffer);
    double val = 0;
    double l1_sum = 0;
    while (infile >> val) {
      vecs[i].push_back(val * factor);
    }
    assert(vecs[i].size() == dim);
    for (size_t j = 0; j < vecs[i].size(); j++) {
      l1_total += std::abs(vecs[i][j]);
      l1_sum += std::abs(vecs[i][j]);
    }
    l1_vec.push_back(l1_sum);
  }
  assert(l1_vec.size() == num_nodes);

  float sum[dim];
  float approx_sum[dim];

  size_t nstep = 1000;
  // dense
  //double start = 32;
  //double end = dim*2;
  // sparse 30%
  double start = 16;
  double end = dim/5.0;

  for (size_t s = 0; s < nstep; s++) {
    int k = start + s * (end - start) / nstep;
    KQuantize(vecs, k, num_nodes, sum, approx_sum, dim);
  }

  return 0;
}


