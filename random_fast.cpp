#include <random>
#include <iostream>

static std::random_device rd;
static std::mt19937 mt(rd());
static std::uniform_real_distribution<double> dist(0.0,1.0);

double uniform_fast() {
  return dist(mt);
}
