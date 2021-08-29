#include "RenderDriverVulkan.h"

#include "vulkan/vulkan_core.h"

VkSampler RD_Vulkan::Core::requestSampler(const SamplerDescription& desc) {
  auto iter = samplers.find(desc);
  if (iter != samplers.end()) {
    return iter->second;
  }
  VkSamplerCreateInfo samplerInfo = {};
  samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
  samplerInfo.magFilter = VK_FILTER_LINEAR;
  samplerInfo.minFilter = VK_FILTER_LINEAR;
  samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
  samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
  samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
  samplerInfo.anisotropyEnable = VK_TRUE;
  samplerInfo.maxAnisotropy = 16;
  samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
  samplerInfo.unnormalizedCoordinates = VK_FALSE;
  samplerInfo.compareEnable = VK_FALSE;
  samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
  samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
  samplerInfo.mipLodBias = 0.0f;
  samplerInfo.minLod = 0;
  samplerInfo.maxLod = desc;

  VK_CHECK_RESULT(vkCreateSampler(device, &samplerInfo, nullptr, &samplers[desc]));
  return samplers[desc];
}

void RD_Vulkan::Core::close() {
  for (auto& [_, sampler] : samplers) {
    vkDestroySampler(device, sampler, nullptr);
  }
  samplers.clear();
}
