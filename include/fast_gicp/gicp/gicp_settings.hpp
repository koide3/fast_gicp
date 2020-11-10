#ifndef FAST_GICP_GICP_SETTINGS_HPP
#define FAST_GICP_GICP_SETTINGS_HPP

namespace fast_gicp {

enum class RegularizationMethod { NONE, MIN_EIG, NORMALIZED_MIN_EIG, PLANE, FROBENIUS };

enum class NeighborSearchMethod { DIRECT27, DIRECT7, DIRECT1 };

enum class VoxelAccumulationMode { ADDITIVE, ADDITIVE_WEIGHTED, MULTIPLICATIVE };
}

#endif