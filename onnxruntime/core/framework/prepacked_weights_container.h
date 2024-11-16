// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/allocator.h"
#include "prepacked_weights.h"

#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>

namespace onnxruntime {

class PrepackedWeightsContainer final {
 public:
  PrepackedWeightsContainer() {
  }

  ~PrepackedWeightsContainer() = default;

  // Returns an allocator keyed by device name.
  // If an allocator doesn't exist for that specific device, an allocator
  // is created and stored in a member to be returned on subsequent calls.
  // Currently, the only supported device is "Cpu".
  AllocatorPtr GetOrCreateAllocator(const std::string& device_name);

  // Returns the PrePackedWeights instance pertaining to the provided key.
  // The key is : op_type + "+" + hash_of_prepacked_buffers_in_the_PrepackedWeights_instance.
  // Throws an exception if the key doesn't exist
  const PrePackedWeights& GetWeight(const std::string& key) const;

  // Writes the PrePackedWeights instance pertaining to the provided key.
  // The key is : op_type + "+" + hash_of_prepacked_buffers_in_the_PrepackedWeights_instance.
  // Returns a boolean indicating if the insertion took place.
  bool WriteWeight(const std::string& key, PrePackedWeights&& packed_weight);

  // Returns a boolean indicating if there is a PrePackedWeights instance
  // pertaining to the provided key.
  // The key is : op_type + "+" + hash_of_prepacked_buffers_in_the_PrepackedWeights_instance.
  bool HasWeight(const std::string& key) const;

  // Returns the number of elements in the container
  size_t GetNumberOfElements() const;

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(PrepackedWeightsContainer);

  // Resource to be acquired by the method that is going to invoke calls to the kernels'
  // PrePack() methods and does the read/write into the pre-packed weights' container.
  // We only want to invoke PrePack() on a kernel that doesn't have a cached version
  // of its pre-packed weight.
  std::mutex mutex_;

  // Define allocators ahead of the container containing tensors because the allocators
  // needs to destructed after the container containing the pre-packed cached tensors
  // because the Tensor buffers will be de-allocated using these allocators
  std::unordered_map<std::string, AllocatorPtr> allocators_;

  // This is an unordered map that holds a mapping between a composite key
  // to PrePackedWeights instances.
  // The key is : op_type + "+" + hash_of_prepacked_buffers_in_the_PrepackedWeights_instance.
  std::unordered_map<std::string, PrePackedWeights> prepacked_weights_map_;
};

/// <summary>
/// This class provides a storage container for PrePackedWeights instances
/// for storing pre-packed weights in the external file.
/// After serialization on disk it can be used to pre-populate shared pre-packed
/// weights if enabled, and also can be used to populate kernels as well.
/// </summary>
class PrepackedWeightsForSerialization final {
 public:
  PrepackedWeightsForSerialization() = default;
  ~PrepackedWeightsForSerialization() = default;

  /// <summary>
  /// Add weight with a key for a given initializer
  /// </summary>
  /// <param name="weight_name"></param>
  /// <param name="key"></param>
  /// <param name="packed_weight"></param>
  /// <returns></returns>
  void WriteWeight(const std::string& weight_name, std::string key, PrePackedWeights&& packed_weight);

  size_t GetBlobNumForWeight(const std::string& weight_name) const;

  const PrePackedWeights& GetBlobForWeight(const std::string& weight_name, size_t index) const;

 private:
  AllocatorPtr cpu_allocator_;

  // Map of key to pre-packed blobs

  using KeyToBlobMap = std::unordered_map<std::string, PrePackedWeights>;
  using KeyToBlobMapIterator = KeyToBlobMap::iterator;

  KeyToBlobMap key_to_blobs_;

  using WeightToPrePacksMap = std::unordered_map<std::string, std::vector<KeyToBlobMapIterator>>;
  WeightToPrePacksMap weight_to_prepacks_;
};

}  // namespace onnxruntime
