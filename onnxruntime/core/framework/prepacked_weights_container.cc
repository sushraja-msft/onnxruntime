// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/prepacked_weights_container.h"
#include "core/framework/allocator_utils.h"

namespace onnxruntime {

AllocatorPtr PrepackedWeightsContainer::GetOrCreateAllocator(const std::string& device_name) {
  auto iter = allocators_.find(device_name);

  if (iter != allocators_.end())
    return iter->second;

  // Support only CPU based allocators for now.
  // as pre-packing is only supported by CPU kernels for now.
  if (device_name == CPU) {
    // TODO: Investigate benefits of using an arena based allocator
    // For now, we go with a non-arena based allocator
    AllocatorCreationInfo device_info{[](int) { return std::make_unique<CPUAllocator>(); },
                                      0, false};
    auto allocator = CreateAllocator(device_info);

    allocators_[device_name] = allocator;

    return allocator;

  } else {
    ORT_THROW("Unsupported device allocator in the context of pre-packed weights caching: ", device_name);
  }
}

const PrePackedWeights& PrepackedWeightsContainer::GetWeight(const std::string& key) const {
  // .at() will throw if the key doesn't exist
  return prepacked_weights_map_.at(key);
}

bool PrepackedWeightsContainer::WriteWeight(const std::string& key, PrePackedWeights&& packed_weight) {
  auto ret = prepacked_weights_map_.insert(std::make_pair(key, std::move(packed_weight)));
  return ret.second;
}

bool PrepackedWeightsContainer::HasWeight(const std::string& key) const {
  return prepacked_weights_map_.find(key) !=
         prepacked_weights_map_.end();
}

size_t PrepackedWeightsContainer::GetNumberOfElements() const {
  return prepacked_weights_map_.size();
}

void PrepackedWeightsForSerialization::WriteWeight(const std::string& weight_name, std::string key,
                                                   PrePackedWeights&& packed_weight) {
  auto result = key_to_blobs_.emplace(std::move(key), std::move(packed_weight));
  weight_to_prepacks_[weight_name].push_back(result.first);
}

size_t PrepackedWeightsForSerialization::GetBlobNumForWeight(const std::string& weight_name) const {
  auto iter = weight_to_prepacks_.find(weight_name);
  if (iter != weight_to_prepacks_.end()) {
    return iter->second.size();
  }
  return 0;
}

const PrePackedWeights& PrepackedWeightsForSerialization::GetBlobForWeight(const std::string& weight_name,
                                                                           size_t index) const {
  auto iter = weight_to_prepacks_.find(weight_name);
  if (iter != weight_to_prepacks_.end()) {
    ORT_ENFORCE(index < iter->second.size(), "Index out of bounds for weight: ", weight_name);
    return iter->second[index]->second;
  }
  ORT_THROW("No prepacked weight found for weight: ", weight_name);
}

}  // namespace onnxruntime
