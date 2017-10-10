//
// Created by Pharrell_WANG on 10/10/2017.
//

#include "TimeCost.h"

double g_dmm1TimeCost = 0;

double getDmm1TimeCost() {
  return g_dmm1TimeCost;
}

void upDmm1TimeCost(double x) {
  g_dmm1TimeCost += x;
}