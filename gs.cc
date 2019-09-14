#include <fstream>
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <random>
#include <unordered_map>
#include <algorithm>


double log2_factorial(int n) {
  double sum = 0;
  for (int i = 1; i <= n; i++) sum += log2(i);
  return sum;
}

double log2_comb(int n, int k) {
  return log2_factorial(n) - log2_factorial(n - k) - log2_factorial(k);
}


void GradSparse(const std::vector< std::vector<double> >& vecs,
                const std::vector< std::vector<double> >& sorted_vecs,
                const double epsilon, const size_t num_nodes,
                float *sum, float *approx_sum, const size_t dim) {
  for (size_t i = 0; i < dim; i++) {
    sum[i] = 0;
    approx_sum[i] = 0;
  }

  double tmp_l1;
  double tmp_l2;
  double slide_l1_l;
  double slide_l2_l;
  double slide_l1_r;
  double slide_l2_r;
  double tmp[num_nodes];
  double topk[num_nodes];

  for (size_t i = 0; i < num_nodes; i++) {
    tmp_l1 = 0;
    tmp_l2 = 0;
    for (size_t j = 0; j < dim; j++) {
      tmp_l1 += sorted_vecs[i][j];
      tmp_l2 += sorted_vecs[i][j]*sorted_vecs[i][j];
    }
    slide_l1_l = 0;
    slide_l1_r = tmp_l1;
    slide_l2_l = 0;
    slide_l2_r = tmp_l2;
    for (size_t j = 0; j < dim; j++) {
      slide_l1_l += sorted_vecs[i][j];
      slide_l1_r -= sorted_vecs[i][j];
      slide_l2_l += sorted_vecs[i][j]*sorted_vecs[i][j];
      slide_l2_r -= sorted_vecs[i][j]*sorted_vecs[i][j];
      topk[i] = sorted_vecs[i].at(j);
      tmp[i] = slide_l1_r / (epsilon * tmp_l2 + slide_l2_r);
      int ind = ((j+1)<dim) ? (j+1) : j;
      if (sorted_vecs[i].at(ind) * tmp[i] < 1) {
        //std::cout << "j " << j/10000. << std::endl;
        break;
      }
    }
  }


  static std::default_random_engine rng;
  std::uniform_real_distribution<double> unif(0, 1);

  double comm_gs = 0;
  double comm_sp_gs = 0;

  for (size_t i = 0; i < num_nodes; i++) {
    const auto& data = vecs[i];
    int table[4];
    table[0] = 0; table[1] = 0; table[2] = 0; table[3] = 0;
    for (size_t j = 0; j < dim; j++) {
      sum[j] += data[j];

      double lambda = tmp[i];
      double p = lambda * std::abs(data[j]);

      if (p >= 1 || std::abs(data[j]) >= topk[i]) {
        comm_gs += 32 + ceil(log2(dim+1));
        comm_sp_gs += 32;
        approx_sum[j] += data[j];
        table[0]++;
      } else if (unif(rng) < p) {
        comm_gs += ceil(log2(dim+1))+1;
        if (data[j] > 0) {
          approx_sum[j] += 1./lambda;
          table[1]++;
        } else if (data[j] < 0) {
          approx_sum[j] += -1./lambda;
          table[2]++;
        } else {
          table[3]++;
        }
      } else {
        table[3]++;
      }
    }
    comm_sp_gs += 32;

    double temp_comm = 0;
    for (size_t tt = 0; tt < 4; tt++) {
      if (table[tt]>0) temp_comm += table[tt]/dim * log2(dim/table[tt]);
    }
    comm_sp_gs += temp_comm * dim + 2;

    comm_gs += 32;
  }


  double err = 0;
  double l2_norm = 0;
  for (size_t i = 0; i < dim; i++) {
    err += (sum[i] - approx_sum[i]) * (sum[i] - approx_sum[i]);
    l2_norm += sum[i] * sum[i];
  }

  std::cout << "comm_GS " << std::min(comm_gs, comm_sp_gs)/num_nodes/dim << std::endl;
  std::cout << "avg_error " << err / dim << std::endl;
}


int main() {
  const size_t num_nodes = 16;
  const size_t dim = 10000;
  std::vector< std::vector<double> > vecs;
  vecs.resize(num_nodes);
  std::vector< std::vector<double> > sorted_vecs;
  sorted_vecs.resize(num_nodes);
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
      sorted_vecs[i].push_back(std::abs(val) * factor);
    }
    assert(vecs[i].size() == dim);
    for (size_t j = 0; j < vecs[i].size(); j++) {
      l1_total += std::abs(vecs[i][j]);
      l1_sum += std::abs(vecs[i][j]);
    }
    l1_vec.push_back(l1_sum);
    std::sort(sorted_vecs[i].begin(), sorted_vecs[i].end());
    std::reverse(sorted_vecs[i].begin(), sorted_vecs[i].end());
  }
  assert(l1_vec.size() == num_nodes);

  float sum[dim];
  float approx_sum[dim];

  size_t nstep = 100;
  double start = 0.000000;
  double end = 1;

  for (size_t s = 0; s < nstep; s++) {
    double epsilon = start + s * (end - start) / nstep;
    GradSparse(vecs, sorted_vecs, epsilon, num_nodes, sum, approx_sum, dim);
  }

  return 0;
}

