// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <optional>
#include <variant>
  
namespace onnxruntime {
// Common holder for potentially different resource accounting
// for different EPs
using ResourceCount = std::variant<size_t>;

/// <summary>
/// This class is used for graph partitioning by EPs
/// It stores the cumulative amount of the resource such as
/// memory that would be consumed by the graph nodes if it is assigned to the EP.
///
/// It provides interfaces to add, remove and query the resource consumption.
///
/// Each provider may assign its own meaning to the resource according to its constraints.
/// </summary>
struct IResourceAccountant {
  virtual ~IResourceAccountant() = default;
  virtual ResourceCount GetConsumedAmount() const = 0;
  virtual void AddConsumedAmount(const ResourceCount& amount) = 0;
  virtual void RemoveConsumedAmount(const ResourceCount& amount) = 0;
  // Absent threshold means auto mode for the EP
  virtual std::optional<ResourceCount> GetThreshold() const = 0;
};

class MemoryAccountant : public IResourceAccountant {
 public:
  MemoryAccountant() = default;
  ~MemoryAccountant() = default;

  explicit MemoryAccountant(size_t threshold)
      : threshold_(threshold) {}

  ResourceCount GetConsumedAmount() const noexcept override {
    return consumed_amount_;
  }

  void AddConsumedAmount(const ResourceCount& amount) noexcept override {
    consumed_amount_ += std::get<0>(amount);
  }

  void RemoveConsumedAmount(const ResourceCount& amount) noexcept override {
    consumed_amount_ -= std::get<0>(amount);
  }

  std::optional<ResourceCount> GetThreshold() const noexcept override {
    return threshold_;
  }

 private:
  size_t consumed_amount_ = 0;
  std::optional<ResourceCount> threshold_;
};

}  // namespace onnxruntime

