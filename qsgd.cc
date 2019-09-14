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

double elias_code(int integer) {
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

void QSGD(const std::vector< std::vector<double> >& vecs, const int k, const size_t num_nodes,
          float *sum, float *approx_sum, const size_t dim) {
  for (size_t i = 0; i < dim; i++) {
    sum[i] = 0;
    approx_sum[i] = 0;
  }

  double s[num_nodes];

  for (size_t i = 0; i < num_nodes; i++) {
    const auto& data = vecs[i];
    s[i] = 0;
    for (size_t j = 0; j < dim; j++) {
      s[i] += data[j] * data[j];
    }
    s[i] = std::sqrt(s[i]);
  }

  static std::default_random_engine rng;
  std::uniform_real_distribution<double> unif(0, 1);

  double comm_QSGD = 0;
  double comm_sp_QSGD = 0;
  int dist = 0;

  for (size_t i = 0; i < num_nodes; i++) {
    const auto& data = vecs[i];
    int nnz = 0;

    for (size_t j = 0; j < dim; j++) {
      double step = s[i] / (k - 1);
      int level = floor(std::abs(data[j])/step);
      double lvalue = level * step;
      double rvalue = (level+1) * step;
      double value = (unif(rng) < (std::abs(data[j]) - lvalue) / step) ? rvalue : lvalue;

      sum[j] += data[j];

      double temp = value;
      if (data[j] < 0) temp = -value;
      approx_sum[j] += temp;

      int l = value == lvalue ? level : (level+1);

      dist++;

      if (l > 0) {
        nnz++;
        comm_sp_QSGD += elias_code(dist);
        comm_sp_QSGD += elias_code(l);
        dist = 0;
      }

      comm_QSGD += elias_code(l+1);
    }
    comm_QSGD += nnz;
    comm_QSGD += 32;

    comm_sp_QSGD += nnz;
    comm_sp_QSGD += 32;
  }

  double err = 0;
  double l2_norm = 0;
  for (size_t i = 0; i < dim; i++) {
    err += (sum[i] - approx_sum[i]) * (sum[i] - approx_sum[i]);
    l2_norm += sum[i] * sum[i];
  }
  std::cout << "comm_QSGD " << std::min(comm_QSGD, comm_sp_QSGD)/num_nodes/dim << std::endl;
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
  //double start = 16;
  //double end = dim;
  // sparse 30%
  double start = 3;
  double end = dim/2.;

  for (size_t s = 0; s < nstep; s++) {
    int k = start + s * (end - start) / nstep;
    QSGD(vecs, k, num_nodes, sum, approx_sum, dim);
  }

  return 0;
}

