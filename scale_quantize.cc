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


int elias_code(int integer) {
  assert(integer > 0);
  int len = 1;
  int tmp = integer;
  for (size_t i = 0; i < 5; i++) {
    if (tmp == 1) break;
    len += ceil(log2(tmp+1));
    tmp = ceil(log2(tmp+1))-1;
  }
  return len;
}

void ScaleQuantize(const std::vector< std::vector<double> >& vecs, const double l1_total,
                    float *sum, float *approx_sum, const size_t dim, double C) {
  for (size_t i = 0; i < dim; i++) {
    sum[i] = 0;
    approx_sum[i] = 0;
  }
  double scale = C / l1_total;
  static std::default_random_engine rng;
  std::uniform_real_distribution<double> unif(0, 1);
  double err = 0;
  std::vector<int> scale_l1_norm;

  double comm_k = 0;
  double comm_sq = 0;

  int nnz = 0;

  for (auto vec : vecs) {
    int l1_sum = 0;
    std::unordered_map<int, int> table;
    nnz = 0;
    for (size_t i = 0; i < dim; i++) {
      int tmp = std::floor(vec[i] * scale + unif(rng));
      int abs_tmp = std::abs(tmp);
      l1_sum += abs_tmp;
      if (abs_tmp != 0) nnz++;

      if (table.count(abs_tmp)) {
        table[abs_tmp]++;
      } else {
        table[abs_tmp] = 1;
      }

      sum[i] += vec[i];
      approx_sum[i] += tmp / scale;
    }
    comm_sq += 64 + nnz + ceil(log2_comb(dim - 1 + l1_sum, l1_sum));

    for (auto item : table) {
      comm_k += elias_code(item.first+1);
      comm_k += elias_code(item.second+1);
    }

    double comm = 0;
    for (auto item : table) {
      comm += item.second/static_cast<double>(dim) * log2(static_cast<double>(dim)/item.second);
    }
    comm *= dim;
    comm_k += comm + nnz + 2 + 64;
  }

  double l2_norm = 0;
  for (size_t i = 0; i < dim; i++) {
    err += (sum[i] - approx_sum[i]) * (sum[i] - approx_sum[i]);
    l2_norm += sum[i] * sum[i];
  }

  const size_t num_nodes = vecs.size();
  std::cout << "comm_SQ " << std::min(comm_sq, comm_k) / num_nodes << std::endl;
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
  double start = dim/5.0;
  double end = num_nodes * dim * 100;
  // sparse 30%
  //double start = dim/10.0;
  //double end = num_nodes * dim * 10;

  for (size_t s = 0; s < nstep; s++) {
    double C = start + s * (end - start) / nstep;
    ScaleQuantize(vecs, l1_total, sum, approx_sum, dim, C);
  }
  return 0;
}

