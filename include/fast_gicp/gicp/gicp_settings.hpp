#ifndef FAST_GICP_GICP_SETTINGS_HPP
#define FAST_GICP_GICP_SETTINGS_HPP

namespace fast_gicp {

enum RegularizationMethod { NONE, MIN_EIG, NORMALIZED_MIN_EIG, PLANE, FROBENIUS };

enum NeighborSearchMethod { DIRECT27, DIRECT7, DIRECT1 };

enum VoxelAccumulationMode { ADDITIVE, ADDITIVE_WEIGHTED, MULTIPLICATIVE };
}

#endif