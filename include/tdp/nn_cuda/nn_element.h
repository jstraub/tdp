#pragma once
#include <tdp/config.h>

namespace tdp {

class NN_Element {
 public:
  TDP_HOST_DEVICE
  NN_Element() : NN_Element(0) {}

  TDP_HOST_DEVICE
  NN_Element(float value) : NN_Element(value, 0) {}

  TDP_HOST_DEVICE
  NN_Element(float value, uint32_t index) : m_value(value), m_index(index) {}

  TDP_HOST_DEVICE
  ~NN_Element() {}

  TDP_HOST_DEVICE
  bool operator< (const NN_Element& e) {
    return m_value < e.m_value;
  }

  TDP_HOST_DEVICE
  bool operator<=(const NN_Element& e) {
    return m_value <= e.m_value;
  }

  TDP_HOST_DEVICE
  bool operator> (const NN_Element& e) {
    return m_value > e.m_value;
  }

  TDP_HOST_DEVICE
  bool operator>=(const NN_Element& e) {
    return m_value >= e.m_value;
  }

  TDP_HOST_DEVICE
  float value() const {
    return m_value;
  }

  float index() const {
    return m_index;
  }

 private:
  float m_value;
  uint32_t m_index;
};

}
