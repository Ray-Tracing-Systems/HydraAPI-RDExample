#include <cassert>

#include <array>
#include <chrono>
#include <set>


#include "RenderDriverVulkan.h"
#include "LiteMath.h"
using namespace HydraLiteMath;

#define GLFW_INCLUDE_VULKAN
#if defined(WIN32)
#include <GLFW/glfw3.h>
#pragma comment(lib, "glfw3dll.lib")
#else
#include <GLFW/glfw3.h>
#endif

#include <iostream>


IHRRenderDriver* CreateVulkan_RenderDriver()
{
  return new RD_Vulkan;
}

#define VK_CHECK_RESULT(f) 													\
{																										\
    VkResult res = (f);															\
    if (res != VK_SUCCESS)													\
    {																								\
        printf("Fatal : VkResult is %d in %s at line %d\n", res,  __FILE__, __LINE__); \
        assert(res == VK_SUCCESS);									\
    }																								\
}

const std::vector<const char*> validationLayers = {
    "VK_LAYER_KHRONOS_validation"
};

const std::vector<const char*> deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

#ifdef NDEBUG
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif

struct Vertex {
  float3 pos;
  float3 color;
  float2 texCoord;

  static VkVertexInputBindingDescription getBindingDescription() {
    VkVertexInputBindingDescription bindingDescription = {};
    bindingDescription.binding = 0;
    bindingDescription.stride = sizeof(Vertex);
    bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    return bindingDescription;
  }

  static std::array<VkVertexInputAttributeDescription, 3> getAttributeDescriptions() {
    std::array<VkVertexInputAttributeDescription, 3> attributeDescriptions = {};

    attributeDescriptions[0].binding = 0;
    attributeDescriptions[0].location = 0;
    attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
    attributeDescriptions[0].offset = offsetof(Vertex, pos);

    attributeDescriptions[1].binding = 0;
    attributeDescriptions[1].location = 1;
    attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
    attributeDescriptions[1].offset = offsetof(Vertex, color);

    attributeDescriptions[2].binding = 0;
    attributeDescriptions[2].location = 2;
    attributeDescriptions[2].format = VK_FORMAT_R32G32_SFLOAT;
    attributeDescriptions[2].offset = offsetof(Vertex, texCoord);

    return attributeDescriptions;
  }
};

RD_Vulkan::QueueFamilyIndices RD_Vulkan::GetQueueFamilyIndex(VkPhysicalDevice physicalDevice)
{
  QueueFamilyIndices indices;
  uint32_t queueFamilyCount;

  vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, NULL);

  // Retrieve all queue families.
  std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
  vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, queueFamilies.data());

  // Now find a family that supports compute.
  uint32_t i = 0;
  for (; i < queueFamilies.size(); ++i)
  {
    VkQueueFamilyProperties props = queueFamilies[i];

    if (props.queueCount > 0 && (props.queueFlags & VK_QUEUE_GRAPHICS_BIT))
    {
      indices.graphicsFamily = i;
    }

    VkBool32 presentSupport = false;
    vkGetPhysicalDeviceSurfaceSupportKHR(physicalDevice, i, surface, &presentSupport);
    if (props.queueCount > 0 && presentSupport) {
      indices.presentFamily = i;
    }
  }

  return indices;
}

void RD_Vulkan::createLogicalDevice()
{
  QueueFamilyIndices indices = GetQueueFamilyIndex(physicalDevice);

  std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
  std::set<uint32_t> uniqueQueueFamilies = { indices.graphicsFamily, indices.presentFamily };

  float queuePriority = 1.0f;
  for (uint32_t queueFamily : uniqueQueueFamilies) {
    VkDeviceQueueCreateInfo queueCreateInfo = {};
    queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueCreateInfo.queueFamilyIndex = queueFamily;
    queueCreateInfo.queueCount = 1;
    queueCreateInfo.pQueuePriorities = &queuePriority;
    queueCreateInfos.push_back(queueCreateInfo);
  }
  VkPhysicalDeviceFeatures deviceFeatures = {};
  deviceFeatures.samplerAnisotropy = VK_TRUE;

  VkDeviceCreateInfo createInfo = {};
  createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;

  createInfo.pQueueCreateInfos = queueCreateInfos.data();
  createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());

  createInfo.pEnabledFeatures = &deviceFeatures;

  createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
  createInfo.ppEnabledExtensionNames = deviceExtensions.data();

  if (enableValidationLayers) {
    createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
    createInfo.ppEnabledLayerNames = validationLayers.data();
  }
  else {
    createInfo.enabledLayerCount = 0;
  }
  VK_CHECK_RESULT(vkCreateDevice(physicalDevice, &createInfo, NULL, &device));

  vkGetDeviceQueue(device, GetQueueFamilyIndex(physicalDevice).graphicsFamily, 0, &graphicsQueue);
}

struct SwapChainSupportDetails {
  VkSurfaceCapabilitiesKHR capabilities;
  std::vector<VkSurfaceFormatKHR> formats;
  std::vector<VkPresentModeKHR> presentModes;
};

SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device, VkSurfaceKHR surface) {
  SwapChainSupportDetails details;

  vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);

  uint32_t formatCount;
  vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);

  if (formatCount != 0) {
    details.formats.resize(formatCount);
    vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.formats.data());
  }

  uint32_t presentModeCount;
  vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);

  if (presentModeCount != 0) {
    details.presentModes.resize(presentModeCount);
    vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, details.presentModes.data());
  }

  return details;
}

void RD_Vulkan::createRenderPass()
{
  VkAttachmentDescription colorAttachment = {};
  colorAttachment.format = swapChainImageFormat;
  colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;

  colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
  colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

  colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;

  colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

  VkAttachmentReference colorAttachmentRef = {};
  colorAttachmentRef.attachment = 0;
  colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

  VkSubpassDescription subpass = {};
  subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
  subpass.colorAttachmentCount = 1;
  subpass.pColorAttachments = &colorAttachmentRef;

  VkSubpassDependency dependency = {};
  dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
  dependency.dstSubpass = 0;
  dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  dependency.srcAccessMask = 0;
  dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

  std::array<VkAttachmentDescription, 1> attachments = { colorAttachment };
  VkRenderPassCreateInfo renderPassInfo = {};
  renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
  renderPassInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
  renderPassInfo.pAttachments = attachments.data();
  renderPassInfo.subpassCount = 1;
  renderPassInfo.pSubpasses = &subpass;
  renderPassInfo.dependencyCount = 1;
  renderPassInfo.pDependencies = &dependency;

  VK_CHECK_RESULT(vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass));
}

static std::vector<char> readFile(const std::string& filename)
{
  std::ifstream file(filename, std::ios::ate | std::ios::binary);

  if (!file.is_open())
  {
    std::string errMsg = std::string("failed to open file ") + filename;
    throw std::runtime_error(errMsg.c_str());
  }

  size_t fileSize = (size_t)file.tellg();
  std::vector<char> buffer(fileSize);

  file.seekg(0);
  file.read(buffer.data(), fileSize);

  file.close();

  return buffer;
}

VkShaderModule createShaderModule(VkDevice device, const std::vector<char>& code) {
  VkShaderModuleCreateInfo createInfo = {};
  createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  createInfo.codeSize = code.size();
  createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

  VkShaderModule shaderModule;
  VK_CHECK_RESULT(vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule));
  return shaderModule;
}

void RD_Vulkan::createGraphicsPipeline() {
  auto vertShaderCode = readFile("shaders/vert.spv");
  auto fragShaderCode = readFile("shaders/frag.spv");

  VkShaderModule vertShaderModule = createShaderModule(device, vertShaderCode);
  VkShaderModule fragShaderModule = createShaderModule(device, fragShaderCode);

  VkPipelineShaderStageCreateInfo vertPipelineShaderStageCreateInfo = {};
  vertPipelineShaderStageCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  vertPipelineShaderStageCreateInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
  vertPipelineShaderStageCreateInfo.module = vertShaderModule;
  vertPipelineShaderStageCreateInfo.pName = "main";

  VkPipelineShaderStageCreateInfo fragPipelineShaderStageCreateInfo = {};
  fragPipelineShaderStageCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  fragPipelineShaderStageCreateInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
  fragPipelineShaderStageCreateInfo.module = fragShaderModule;
  fragPipelineShaderStageCreateInfo.pName = "main";

  std::array<VkPipelineShaderStageCreateInfo, 2> stages = { vertPipelineShaderStageCreateInfo, fragPipelineShaderStageCreateInfo };

  auto bindingDescription = Vertex::getBindingDescription();
  auto attributeDescriptions = Vertex::getAttributeDescriptions();

  VkPipelineVertexInputStateCreateInfo vertexInputStateCreateInfo = {};
  vertexInputStateCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
  vertexInputStateCreateInfo.vertexBindingDescriptionCount = 1;
  vertexInputStateCreateInfo.pVertexBindingDescriptions = &bindingDescription;
  vertexInputStateCreateInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
  vertexInputStateCreateInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

  VkPipelineInputAssemblyStateCreateInfo inputAssemblyDesc = {};
  inputAssemblyDesc.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
  inputAssemblyDesc.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
  inputAssemblyDesc.primitiveRestartEnable = VK_FALSE;

  VkViewport viewport = {};
  viewport.x = 0;
  viewport.y = 0;
  viewport.width = static_cast<float>(swapChainExtent.width);
  viewport.height = static_cast<float>(swapChainExtent.height);
  viewport.minDepth = 0;
  viewport.maxDepth = 1;

  VkRect2D scissor = {};
  scissor.offset.x = scissor.offset.y = 0;
  scissor.extent = swapChainExtent;

  VkPipelineViewportStateCreateInfo viewportState = {};
  viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
  viewportState.viewportCount = 1;
  viewportState.pViewports = &viewport;
  viewportState.scissorCount = 1;
  viewportState.pScissors = &scissor;

  VkPipelineRasterizationStateCreateInfo rasterizationState = {};
  rasterizationState.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
  rasterizationState.depthClampEnable = VK_FALSE;
  rasterizationState.lineWidth = 1;
  rasterizationState.rasterizerDiscardEnable = VK_FALSE;
  rasterizationState.cullMode = VK_CULL_MODE_NONE;
  rasterizationState.frontFace = VK_FRONT_FACE_CLOCKWISE;
  rasterizationState.depthBiasEnable = VK_FALSE;
  rasterizationState.depthBiasConstantFactor = 0.0f; // Optional
  rasterizationState.depthBiasClamp = 0.0f; // Optional
  rasterizationState.depthBiasSlopeFactor = 0.0f; // Optional

  VkPipelineMultisampleStateCreateInfo multisampling = {};
  multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
  multisampling.sampleShadingEnable = VK_FALSE;
  multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
  multisampling.minSampleShading = 1.0f; // Optional
  multisampling.pSampleMask = nullptr; // Optional
  multisampling.alphaToCoverageEnable = VK_FALSE; // Optional
  multisampling.alphaToOneEnable = VK_FALSE; // Optional

  VkPipelineColorBlendAttachmentState blendStateAttach = {};
  blendStateAttach.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
  blendStateAttach.blendEnable = VK_FALSE;
  blendStateAttach.srcColorBlendFactor = VK_BLEND_FACTOR_ONE; // Optional
  blendStateAttach.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO; // Optional
  blendStateAttach.colorBlendOp = VK_BLEND_OP_ADD; // Optional
  blendStateAttach.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE; // Optional
  blendStateAttach.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO; // Optional
  blendStateAttach.alphaBlendOp = VK_BLEND_OP_ADD; // Optional

  VkPipelineColorBlendStateCreateInfo colorBlending = {};
  colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
  colorBlending.logicOpEnable = VK_FALSE;
  colorBlending.logicOp = VK_LOGIC_OP_COPY; // Optional
  colorBlending.attachmentCount = 1;
  colorBlending.pAttachments = &blendStateAttach;
  colorBlending.blendConstants[0] = 0.0f; // Optional
  colorBlending.blendConstants[1] = 0.0f; // Optional
  colorBlending.blendConstants[2] = 0.0f; // Optional
  colorBlending.blendConstants[3] = 0.0f; // Optional

  VkDynamicState dynamicStates[] = {
    VK_DYNAMIC_STATE_VIEWPORT,
    VK_DYNAMIC_STATE_LINE_WIDTH
  };

  VkPipelineDynamicStateCreateInfo dynamicState = {};
  dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
  dynamicState.dynamicStateCount = 2;
  dynamicState.pDynamicStates = dynamicStates;

  VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = {};
  pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pipelineLayoutCreateInfo.setLayoutCount = 1;
  pipelineLayoutCreateInfo.pSetLayouts = &descriptorSetLayout;
  pipelineLayoutCreateInfo.pushConstantRangeCount = 0; // Optional
  pipelineLayoutCreateInfo.pPushConstantRanges = nullptr; // Optional

  VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, nullptr, &pipelineLayout));

  VkGraphicsPipelineCreateInfo graphicsPipelineCreateInfo = {};
  graphicsPipelineCreateInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
  graphicsPipelineCreateInfo.stageCount = static_cast<uint32_t>(stages.size());
  graphicsPipelineCreateInfo.pStages = stages.data();
  graphicsPipelineCreateInfo.pVertexInputState = &vertexInputStateCreateInfo;
  graphicsPipelineCreateInfo.pInputAssemblyState = &inputAssemblyDesc;
  graphicsPipelineCreateInfo.pViewportState = &viewportState;
  graphicsPipelineCreateInfo.pRasterizationState = &rasterizationState;
  graphicsPipelineCreateInfo.pMultisampleState = &multisampling;
  graphicsPipelineCreateInfo.pDepthStencilState = nullptr; // Optional
  graphicsPipelineCreateInfo.pColorBlendState = &colorBlending;
  graphicsPipelineCreateInfo.pDynamicState = nullptr; // Optional
  graphicsPipelineCreateInfo.layout = pipelineLayout;
  graphicsPipelineCreateInfo.renderPass = renderPass;
  graphicsPipelineCreateInfo.subpass = 0;
  graphicsPipelineCreateInfo.basePipelineHandle = VK_NULL_HANDLE;
  graphicsPipelineCreateInfo.basePipelineIndex = -1; // Optional

  VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &graphicsPipelineCreateInfo, nullptr, &graphicsPipeline));

  vkDestroyShaderModule(device, fragShaderModule, nullptr);
  vkDestroyShaderModule(device, vertShaderModule, nullptr);
}

uint32_t findMemoryType(VkPhysicalDevice physicalDevice, uint32_t typeFilter, VkMemoryPropertyFlags properties) {
  VkPhysicalDeviceMemoryProperties memProperties;
  vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

  for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
    if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
      return i;
    }
  }

  throw std::runtime_error("failed to find suitable memory type!");
}

void RD_Vulkan::BufferManager::createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory)
{
  VkBufferCreateInfo bufferInfo = {};
  bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bufferInfo.size = size;
  bufferInfo.usage = usage;
  bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  VK_CHECK_RESULT(vkCreateBuffer(deviceRef, &bufferInfo, nullptr, &buffer));

  VkMemoryRequirements memRequirements;
  vkGetBufferMemoryRequirements(deviceRef, buffer, &memRequirements);

  VkMemoryAllocateInfo allocInfo = {};
  allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  allocInfo.allocationSize = memRequirements.size;
  allocInfo.memoryTypeIndex = findMemoryType(physicalDeviceRef, memRequirements.memoryTypeBits, properties);

  VK_CHECK_RESULT(vkAllocateMemory(deviceRef, &allocInfo, nullptr, &bufferMemory));

  vkBindBufferMemory(deviceRef, buffer, bufferMemory, 0);
}

void RD_Vulkan::Mesh::createVertexBuffer(const std::vector<Vertex>& vertices) {
  VkDeviceSize bufferSize = sizeof(vertices[0]) * vertices.size();

  VkBuffer stagingBuffer;
  VkDeviceMemory stagingBufferMemory;
  BufferManager::get().createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

  void* data;
  vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
  memcpy(data, vertices.data(), (size_t)bufferSize);
  vkUnmapMemory(device, stagingBufferMemory);

  BufferManager::get().createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, vertexBuffer, vertexBufferMemory);

  BufferManager::get().copyBuffer(stagingBuffer, vertexBuffer, bufferSize);
  vkDestroyBuffer(device, stagingBuffer, nullptr);
  vkFreeMemory(device, stagingBufferMemory, nullptr);
}

void RD_Vulkan::BufferManager::copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size) {
  SingleTimeCommandsContext context;

  VkBufferCopy copyRegion = {};
  copyRegion.size = size;
  vkCmdCopyBuffer(context.getCB(), srcBuffer, dstBuffer, 1, &copyRegion);
}

void RD_Vulkan::BufferManager::transitionImageLayout(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout) {
  SingleTimeCommandsContext context;

  VkImageMemoryBarrier barrier = {};
  barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  barrier.oldLayout = oldLayout;
  barrier.newLayout = newLayout;
  barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.image = image;
  barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  barrier.subresourceRange.baseMipLevel = 0;
  barrier.subresourceRange.levelCount = 1;
  barrier.subresourceRange.baseArrayLayer = 0;
  barrier.subresourceRange.layerCount = 1;
  barrier.srcAccessMask = 0; // TODO
  barrier.dstAccessMask = 0; // TODO

  VkPipelineStageFlags sourceStage;
  VkPipelineStageFlags destinationStage;

  if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
    barrier.srcAccessMask = 0;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

    sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
  }
  else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
  }
  else {
    throw std::invalid_argument("unsupported layout transition!");
  }

  vkCmdPipelineBarrier(
    context.getCB(),
    sourceStage, destinationStage,
    0,
    0, nullptr,
    0, nullptr,
    1, &barrier
  );
}

void RD_Vulkan::BufferManager::copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height) {
  SingleTimeCommandsContext context;

  VkBufferImageCopy region = {};
  region.bufferOffset = 0;
  region.bufferRowLength = 0;
  region.bufferImageHeight = 0;

  region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  region.imageSubresource.mipLevel = 0;
  region.imageSubresource.baseArrayLayer = 0;
  region.imageSubresource.layerCount = 1;

  region.imageOffset = { 0, 0, 0 };
  region.imageExtent = {
      width,
      height,
      1
  };

  vkCmdCopyBufferToImage(
    context.getCB(),
    buffer,
    image,
    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
    1,
    &region
  );
}

// Find a memory in `memoryTypeBitsRequirement` that includes all of `requiredProperties`
int32_t findProperties(const VkPhysicalDeviceMemoryProperties* pMemoryProperties,
  uint32_t memoryTypeBitsRequirement,
  VkMemoryPropertyFlags requiredProperties) {
  const uint32_t memoryCount = pMemoryProperties->memoryTypeCount;
  for (uint32_t memoryIndex = 0; memoryIndex < memoryCount; ++memoryIndex) {
    const uint32_t memoryTypeBits = (1 << memoryIndex);
    const bool isRequiredMemoryType = (memoryTypeBitsRequirement & memoryTypeBits) != 0;

    const VkMemoryPropertyFlags properties =
      pMemoryProperties->memoryTypes[memoryIndex].propertyFlags;
    const bool hasRequiredProperties =
      (properties & requiredProperties) == requiredProperties;

    if (isRequiredMemoryType && hasRequiredProperties)
      return static_cast<int32_t>(memoryIndex);
  }

  // failed to find memory type
  return -1;
}

extern GLFWwindow* g_window;

bool checkValidationLayerSupport() {
  uint32_t layerCount;
  vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

  std::vector<VkLayerProperties> availableLayers(layerCount);
  vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

  for (const char* layerName : validationLayers) {
    bool layerFound = false;

    for (const auto& layerProperties : availableLayers) {
      if (strcmp(layerName, layerProperties.layerName) == 0) {
        layerFound = true;
        break;
      }
    }

    if (!layerFound) {
      return false;
    }
  }

  return true;
}

void RD_Vulkan::createInstance()
{
  if (enableValidationLayers && !checkValidationLayerSupport()) {
    throw std::runtime_error("validation layers requested, but not available!");
  }

  VkApplicationInfo appInfo = {};
  appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  appInfo.pApplicationName = "Hello Triangle";
  appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
  appInfo.pEngineName = "No Engine";
  appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
  appInfo.apiVersion = VK_API_VERSION_1_0;

  VkInstanceCreateInfo createInfo = {};
  createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  createInfo.pApplicationInfo = &appInfo;

  uint32_t extensionCount = 0;
  vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);
  std::vector<VkExtensionProperties> extensions(extensionCount);
  vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, extensions.data());
  std::cout << "available extensions:" << std::endl;

  for (const auto& extension : extensions) {
    std::cout << "\t" << extension.extensionName << std::endl;
  }

  uint32_t glfwExtensionCount = 0;
  const char** glfwExtensions;

  glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
  for (uint32_t i = 0; i < glfwExtensionCount; ++i) {
    bool found = false;
    for (uint32_t j = 0; j < extensionCount && !found; ++j) {
      found = strncmp(extensions[j].extensionName, glfwExtensions[i], 256) == 0;
    }
    assert(found);
  }

  createInfo.enabledExtensionCount = glfwExtensionCount;
  createInfo.ppEnabledExtensionNames = glfwExtensions;

  if (enableValidationLayers) {
    createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
    createInfo.ppEnabledLayerNames = validationLayers.data();
  } else {
    createInfo.enabledLayerCount = 0;
  }

  VK_CHECK_RESULT(vkCreateInstance(&createInfo, nullptr, &vk_inst));
}

bool checkDeviceExtensionSupport(VkPhysicalDevice device) {
  uint32_t extensionCount;
  vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);

  std::vector<VkExtensionProperties> availableExtensions(extensionCount);
  vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());

  std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());

  for (const auto& extension : availableExtensions) {
    requiredExtensions.erase(extension.extensionName);
  }

  return requiredExtensions.empty();
}

bool isDeviceSuitable(VkPhysicalDevice device, VkSurfaceKHR surface) {
  bool extensionsSupported = checkDeviceExtensionSupport(device);

  bool swapChainAdequate = false;
  if (extensionsSupported) {
    SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device, surface);
    swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
  }

  VkPhysicalDeviceFeatures supportedFeatures;
  vkGetPhysicalDeviceFeatures(device, &supportedFeatures);

  return extensionsSupported && swapChainAdequate && supportedFeatures.samplerAnisotropy;
}

void RD_Vulkan::pickPhysicalDevice() {
  uint32_t physicalDeviceCount;

  VK_CHECK_RESULT(vkEnumeratePhysicalDevices(vk_inst, &physicalDeviceCount, nullptr));
  if (physicalDeviceCount == 0) {
    throw std::runtime_error("failed to find GPUs with Vulkan support!");
  }
  std::cout << physicalDeviceCount << " physical devices found" << std::endl;
  std::vector<VkPhysicalDevice> physDevices(physicalDeviceCount);
  VK_CHECK_RESULT(vkEnumeratePhysicalDevices(vk_inst, &physicalDeviceCount, physDevices.data()));

  std::cout << "DEVICES:" << std::endl;
  for (uint32_t i = 0; i < physicalDeviceCount; ++i)
  {
    VkPhysicalDeviceProperties prop;
    vkGetPhysicalDeviceProperties(physDevices[i], &prop);
    std::cout << "Device name: " << prop.deviceName << std::endl;
  }

  for (const auto& device : physDevices) {
    if (isDeviceSuitable(device, surface)) {
      physicalDevice = device;
      break;
    }
  }

  if (physicalDevice == VK_NULL_HANDLE) {
    throw std::runtime_error("failed to find a suitable GPU!");
  }
}

void RD_Vulkan::createSurface() {
  VK_CHECK_RESULT(glfwCreateWindowSurface(vk_inst, g_window, nullptr, &surface));
}

VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats) {
  for (const auto& availableFormat : availableFormats) {
    if (availableFormat.format == VK_FORMAT_B8G8R8A8_UNORM && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
      return availableFormat;
    }
  }
  return availableFormats[0];
}

VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes) {
  for (const auto& availablePresentMode : availablePresentModes) {
    if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR) {
      return availablePresentMode;
    }
  }
  return VK_PRESENT_MODE_FIFO_KHR;
}

VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities) {
  if (capabilities.currentExtent.width != UINT32_MAX) {
    return capabilities.currentExtent;
  } else {
    int width, height;
    glfwGetFramebufferSize(g_window, &width, &height);

    VkExtent2D actualExtent = {
        static_cast<uint32_t>(width),
        static_cast<uint32_t>(height)
    };

    actualExtent.width = std::max(capabilities.minImageExtent.width, std::min(capabilities.maxImageExtent.width, actualExtent.width));
    actualExtent.height = std::max(capabilities.minImageExtent.height, std::min(capabilities.maxImageExtent.height, actualExtent.height));

    return actualExtent;
  }
}

void RD_Vulkan::createSwapChain() {
  SwapChainSupportDetails swapChainSupport = querySwapChainSupport(physicalDevice, surface);

  VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
  VkPresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
  VkExtent2D extent = chooseSwapExtent(swapChainSupport.capabilities);

  uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
  if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount) {
    imageCount = swapChainSupport.capabilities.maxImageCount;
  }

  VkSwapchainCreateInfoKHR createInfo = {};
  createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
  createInfo.surface = surface;
  createInfo.minImageCount = imageCount;
  createInfo.imageFormat = surfaceFormat.format;
  createInfo.imageColorSpace = surfaceFormat.colorSpace;
  createInfo.imageExtent = extent;
  createInfo.imageArrayLayers = 1;
  createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

  QueueFamilyIndices indices = GetQueueFamilyIndex(physicalDevice);
  uint32_t queueFamilyIndices[] = { indices.graphicsFamily, indices.presentFamily };

  if (indices.graphicsFamily != indices.presentFamily) {
    createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
    createInfo.queueFamilyIndexCount = 2;
    createInfo.pQueueFamilyIndices = queueFamilyIndices;
  }
  else {
    createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    createInfo.queueFamilyIndexCount = 0; // Optional
    createInfo.pQueueFamilyIndices = nullptr; // Optional
  }

  createInfo.preTransform = swapChainSupport.capabilities.currentTransform;
  createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
  createInfo.presentMode = presentMode;
  createInfo.clipped = VK_TRUE;
  createInfo.oldSwapchain = VK_NULL_HANDLE;
  VK_CHECK_RESULT(vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapchain));

  vkGetSwapchainImagesKHR(device, swapchain, &imageCount, nullptr);
  swapChainImages.resize(imageCount);
  vkGetSwapchainImagesKHR(device, swapchain, &imageCount, swapChainImages.data());

  swapChainImageFormat = surfaceFormat.format;
  swapChainExtent = extent;
}

void RD_Vulkan::createImageViews() {
  swapChainImageViews.resize(swapChainImages.size());

  for (int i = 0; i < swapChainImageViews.size(); ++i) {
    swapChainImageViews[i] = BufferManager::get().createImageView(swapChainImages[i], swapChainImageFormat);
  }
}

void RD_Vulkan::createFramebuffers() {
  swapChainFramebuffers.resize(swapChainImageViews.size());
  for (int i = 0; i < swapChainImages.size(); ++i)
  {
    VkFramebufferCreateInfo framebufferCreateInfo = {};
    framebufferCreateInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    framebufferCreateInfo.attachmentCount = 1;
    framebufferCreateInfo.pAttachments = &swapChainImageViews[i];
    framebufferCreateInfo.width = swapChainExtent.width;
    framebufferCreateInfo.height = swapChainExtent.height;
    framebufferCreateInfo.layers = 1;
    framebufferCreateInfo.renderPass = renderPass;
    VK_CHECK_RESULT(vkCreateFramebuffer(device, &framebufferCreateInfo, nullptr, &swapChainFramebuffers[i]));
  }
}

void RD_Vulkan::createCommandPool() {
  VkCommandPoolCreateInfo commandPoolCreateInfo;
  commandPoolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  commandPoolCreateInfo.queueFamilyIndex = GetQueueFamilyIndex(physicalDevice).graphicsFamily;
  commandPoolCreateInfo.flags = 0;
  VK_CHECK_RESULT(vkCreateCommandPool(device, &commandPoolCreateInfo, nullptr, &commandPool));
}

void RD_Vulkan::Mesh::bind(VkCommandBuffer command_buffer) {
  VkBuffer vertexBuffers[] = { vertexBuffer };
  VkDeviceSize offsets[] = { 0 };
  vkCmdBindVertexBuffers(command_buffer, 0, 1, vertexBuffers, offsets);

  vkCmdBindIndexBuffer(command_buffer, indexBuffer, 0, (indicesCount >= 1 << 16) ? VK_INDEX_TYPE_UINT32 : VK_INDEX_TYPE_UINT16);
}

void RD_Vulkan::InstancesCollection::bind(VkCommandBuffer command_buffer, VkPipelineLayout layout, int idx) {
  vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, 1, &descriptorSets[idx], 0, nullptr);
}

void RD_Vulkan::BufferManager::createImage(uint32_t width, uint32_t height, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage,
  VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& imageMemory)
{
  VkImageCreateInfo imageInfo = {};
  imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  imageInfo.imageType = VK_IMAGE_TYPE_2D;
  imageInfo.extent.width = width;
  imageInfo.extent.height = height;
  imageInfo.extent.depth = 1;
  imageInfo.mipLevels = 1;
  imageInfo.arrayLayers = 1;
  imageInfo.format = format;
  imageInfo.tiling = tiling;
  imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  imageInfo.usage = usage;
  imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
  imageInfo.flags = 0; // Optional
  VK_CHECK_RESULT(vkCreateImage(deviceRef, &imageInfo, nullptr, &image));

  VkMemoryRequirements memRequirements;
  vkGetImageMemoryRequirements(deviceRef, image, &memRequirements);

  VkMemoryAllocateInfo allocInfo = {};
  allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  allocInfo.allocationSize = memRequirements.size;
  allocInfo.memoryTypeIndex = findMemoryType(physicalDeviceRef, memRequirements.memoryTypeBits, properties);
  VK_CHECK_RESULT(vkAllocateMemory(deviceRef, &allocInfo, nullptr, &imageMemory));

  vkBindImageMemory(deviceRef, image, imageMemory, 0);
}

VkCommandBuffer RD_Vulkan::BufferManager::beginSingleTimeCommands() {
  VkCommandBufferAllocateInfo allocInfo = {};
  allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  allocInfo.commandPool = commandPoolRef;
  allocInfo.commandBufferCount = 1;

  VkCommandBuffer commandBuffer;
  vkAllocateCommandBuffers(deviceRef, &allocInfo, &commandBuffer);

  VkCommandBufferBeginInfo beginInfo = {};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

  vkBeginCommandBuffer(commandBuffer, &beginInfo);

  return commandBuffer;
}

void RD_Vulkan::BufferManager::endSingleTimeCommands(VkCommandBuffer commandBuffer) {
  vkEndCommandBuffer(commandBuffer);

  VkSubmitInfo submitInfo = {};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &commandBuffer;

  vkQueueSubmit(queueRef, 1, &submitInfo, VK_NULL_HANDLE);
  vkQueueWaitIdle(queueRef);

  vkFreeCommandBuffers(deviceRef, commandPoolRef, 1, &commandBuffer);
}

void RD_Vulkan::Texture::createTextureImage(uint32_t width, uint32_t height, const uint8_t *image) {
  VkDeviceSize imageSize = width * height * 4;

  VkBuffer stagingBuffer;
  VkDeviceMemory stagingBufferMemory;

  BufferManager::get().createBuffer(imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

  void* data;
  vkMapMemory(deviceRef, stagingBufferMemory, 0, imageSize, 0, &data);
  memcpy(data, image, static_cast<size_t>(imageSize));
  vkUnmapMemory(deviceRef, stagingBufferMemory);

  BufferManager::get().createImage(width, height, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, textureImage, textureImageMemory);

  BufferManager::get().transitionImageLayout(textureImage, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
  BufferManager::get().copyBufferToImage(stagingBuffer, textureImage, width, height);
  BufferManager::get().transitionImageLayout(textureImage, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

  vkDestroyBuffer(deviceRef, stagingBuffer, nullptr);
  vkFreeMemory(deviceRef, stagingBufferMemory, nullptr);
}

VkImageView RD_Vulkan::BufferManager::createImageView(VkImage image, VkFormat format) {
  VkImageViewCreateInfo viewInfo = {};
  viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  viewInfo.image = image;
  viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
  viewInfo.format = format;
  viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  viewInfo.subresourceRange.baseMipLevel = 0;
  viewInfo.subresourceRange.levelCount = 1;
  viewInfo.subresourceRange.baseArrayLayer = 0;
  viewInfo.subresourceRange.layerCount = 1;

  VkImageView imageView;
  VK_CHECK_RESULT(vkCreateImageView(deviceRef, &viewInfo, nullptr, &imageView));

  return imageView;
}

void RD_Vulkan::Texture::createTextureImageView() {
  textureImageView = BufferManager::get().createImageView(textureImage, VK_FORMAT_R8G8B8A8_UNORM);
}

void RD_Vulkan::createCommandBuffers() {
  commandBuffers.resize(swapChainImages.size());

  VkCommandBufferAllocateInfo allocInfo = {};
  allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  allocInfo.commandPool = commandPool;
  allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  allocInfo.commandBufferCount = static_cast<uint32_t>(commandBuffers.size());

  VK_CHECK_RESULT(vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data()));

  for (int i = 0; i < commandBuffers.size(); ++i)
  {
    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = 0;
    beginInfo.pInheritanceInfo = nullptr;
    VK_CHECK_RESULT(vkBeginCommandBuffer(commandBuffers[i], &beginInfo));

    VkRenderPassBeginInfo renderPassBeginInfo = {};
    renderPassBeginInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    renderPassBeginInfo.renderPass = renderPass;
    renderPassBeginInfo.framebuffer = swapChainFramebuffers[i];
    renderPassBeginInfo.renderArea.offset = {0, 0};
    renderPassBeginInfo.renderArea.extent = swapChainExtent;
    VkClearValue clearValue = {};
    clearValue.color = { 0.0f, 0.0f, 0.0f, 1.0f };
    renderPassBeginInfo.clearValueCount = 1;
    renderPassBeginInfo.pClearValues = &clearValue;
    vkCmdBeginRenderPass(commandBuffers[i], &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

    vkCmdBindPipeline(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);

    for (auto &instance : instances) {
      for (uint32_t j = 0; j < meshes[instance->getMeshId()].getMeshesCount(); ++j) {
        Mesh* mesh = meshes[instance->getMeshId()].getMesh(j);
        mesh->bind(commandBuffers[i]);
        instance->bind(commandBuffers[i], pipelineLayout, i);
        vkCmdDrawIndexed(commandBuffers[i], mesh->getIndicesCount(), 1, 0, 0, 0);
      }
    }
    vkCmdEndRenderPass(commandBuffers[i]);
    VK_CHECK_RESULT(vkEndCommandBuffer(commandBuffers[i]));
  }
}

void RD_Vulkan::createSyncObjects() {
  VkSemaphoreCreateInfo semaphoreInfo = {};
  semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

  VkFenceCreateInfo fenceInfo = {};
  fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

  for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
    VK_CHECK_RESULT(vkCreateSemaphore(device, &semaphoreInfo, nullptr, &imageAvailableSemaphores[i]));
    VK_CHECK_RESULT(vkCreateSemaphore(device, &semaphoreInfo, nullptr, &renderFinishedSemaphores[i]));
    VK_CHECK_RESULT(vkCreateFence(device, &fenceInfo, nullptr, &inFlightFences[i]));
  }
}

RD_Vulkan::InstancesCollection::~InstancesCollection() {
  for (size_t i = 0; i < swapchain_images; i++) {
    vkDestroyBuffer(device, uniformBuffers[i], nullptr);
    vkFreeMemory(device, uniformBuffersMemory[i], nullptr);
  }
  vkDestroyDescriptorPool(device, descriptorPool, nullptr);
}

void RD_Vulkan::cleanupSwapChain() {
  for (auto framebuffer : swapChainFramebuffers) {
    vkDestroyFramebuffer(device, framebuffer, nullptr);
  }
  vkFreeCommandBuffers(device, commandPool, static_cast<uint32_t>(commandBuffers.size()), commandBuffers.data());

  vkDestroyPipeline(device, graphicsPipeline, nullptr);
  vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
  vkDestroyRenderPass(device, renderPass, nullptr);

  for (auto imageView : swapChainImageViews) {
    vkDestroyImageView(device, imageView, nullptr);
  }

  vkDestroySwapchainKHR(device, swapchain, nullptr);

  instances.clear();
}

void RD_Vulkan::recreateSwapChain() {
  vkDeviceWaitIdle(device);

  cleanupSwapChain();

  createSwapChain();
  createImageViews();
  createRenderPass();
  createGraphicsPipeline();
  createFramebuffers();
  createCommandBuffers();
}

void RD_Vulkan::createDescriptorSetLayout() {
  VkDescriptorSetLayoutBinding uboLayoutBinding = {};
  uboLayoutBinding.binding = 0;
  uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  uboLayoutBinding.descriptorCount = 1;
  uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
  uboLayoutBinding.pImmutableSamplers = nullptr; // Optional

  VkDescriptorSetLayoutBinding samplerLayoutBinding = {};
  samplerLayoutBinding.binding = 1;
  samplerLayoutBinding.descriptorCount = 1;
  samplerLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  samplerLayoutBinding.pImmutableSamplers = nullptr;
  samplerLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

  std::array<VkDescriptorSetLayoutBinding, 2> bindings = { uboLayoutBinding, samplerLayoutBinding };
  VkDescriptorSetLayoutCreateInfo layoutInfo = {};
  layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
  layoutInfo.pBindings = bindings.data();

  VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout));
}

void RD_Vulkan::InstancesCollection::createUniformBuffers() {
  VkDeviceSize bufferSize = sizeof(float4x4) * MAX_INSTANCES;

  uniformBuffers.resize(swapchain_images);
  uniformBuffersMemory.resize(swapchain_images);

  for (size_t i = 0; i < swapchain_images; i++) {
    BufferManager::get().createBuffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, uniformBuffers[i], uniformBuffersMemory[i]);
  }
}

void RD_Vulkan::InstancesCollection::createDescriptorPool() {
  std::array<VkDescriptorPoolSize, 2> poolSizes = {};
  poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  poolSizes[0].descriptorCount = static_cast<uint32_t>(swapchain_images);
  poolSizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  poolSizes[1].descriptorCount = static_cast<uint32_t>(swapchain_images);

  VkDescriptorPoolCreateInfo poolInfo = {};
  poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
  poolInfo.pPoolSizes = poolSizes.data();
  poolInfo.maxSets = static_cast<uint32_t>(swapchain_images);

  VK_CHECK_RESULT(vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool));
}

VkDescriptorImageInfo RD_Vulkan::Texture::getDescriptor() const {
  VkDescriptorImageInfo imageInfo = {};
  imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  imageInfo.imageView = textureImageView;
  imageInfo.sampler = textureSampler;
  return imageInfo;
}

void RD_Vulkan::InstancesCollection::createDescriptorSets() {
  std::vector<VkDescriptorSetLayout> layouts(swapchain_images, descriptorSetLayout);
  VkDescriptorSetAllocateInfo allocInfo = {};
  allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  allocInfo.descriptorPool = descriptorPool;
  allocInfo.descriptorSetCount = static_cast<uint32_t>(swapchain_images);
  allocInfo.pSetLayouts = layouts.data();

  descriptorSets.resize(swapchain_images);
  VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, descriptorSets.data()));

  for (size_t i = 0; i < swapchain_images; i++) {
    VkDescriptorBufferInfo bufferInfo = {};
    bufferInfo.buffer = uniformBuffers[i];
    bufferInfo.offset = 0;
    bufferInfo.range = sizeof(float4x4) * MAX_INSTANCES;

    VkDescriptorImageInfo imageInfo;

    std::vector<VkWriteDescriptorSet> descriptorWrites(1);
    descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[0].dstSet = descriptorSets[i];
    descriptorWrites[0].dstBinding = 0;
    descriptorWrites[0].dstArrayElement = 0;
    descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    descriptorWrites[0].descriptorCount = 1;
    descriptorWrites[0].pBufferInfo = &bufferInfo;
    descriptorWrites[0].pImageInfo = nullptr; // Optional
    descriptorWrites[0].pTexelBufferView = nullptr; // Optional

    if (texture) {
      descriptorWrites.resize(2);
      imageInfo = texture->getDescriptor();
      descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      descriptorWrites[1].dstSet = descriptorSets[i];
      descriptorWrites[1].dstBinding = 1;
      descriptorWrites[1].dstArrayElement = 0;
      descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
      descriptorWrites[1].descriptorCount = 1;
      descriptorWrites[1].pImageInfo = &imageInfo;
    }

    vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
  }
}

void RD_Vulkan::Texture::createTextureSampler() {
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
  samplerInfo.minLod = 0.0f;
  samplerInfo.maxLod = 0.0f;

  VK_CHECK_RESULT(vkCreateSampler(deviceRef, &samplerInfo, nullptr, &textureSampler));
}

void RD_Vulkan::createDefaultTexture() {
  const uint8_t WHITE_COLOR[4] = {0xFF, 0xFF, 0xFF, 0xFF};
  defaultTexture = std::make_unique<Texture>(device, 1, 1, WHITE_COLOR);
}

RD_Vulkan::RD_Vulkan()
{
  createInstance();
  createSurface();
  pickPhysicalDevice();
  createLogicalDevice();
  createSwapChain();
  BufferManager::get().init(physicalDevice, device, commandPool, graphicsQueue);
  createImageViews();
  createRenderPass();
  createDescriptorSetLayout();
  createGraphicsPipeline();
  createFramebuffers();
  createCommandPool();
  BufferManager::get().init(physicalDevice, device, commandPool, graphicsQueue);
  createDefaultTexture();
  createCommandBuffers();
  createSyncObjects();

  camFov       = 45.0f;
  camNearPlane = 0.1f;
  camFarPlane  = 1000.0f;

  camPos[0]    = 0.0f; camPos[1]    = 0.0f; camPos[2]    = 0.0f;
  camLookAt[0] = 0.0f; camLookAt[1] = 0.0f; camLookAt[2] = -1.0f;

  m_width  = 1024;
  m_height = 1024;
  camUseMatrices = false;
}

RD_Vulkan::Texture::~Texture() {
  vkDestroySampler(deviceRef, textureSampler, nullptr);
  vkDestroyImageView(deviceRef, textureImageView, nullptr);

  vkDestroyImage(deviceRef, textureImage, nullptr);
  vkFreeMemory(deviceRef, textureImageMemory, nullptr);
}

RD_Vulkan::~RD_Vulkan()
{
  vkDeviceWaitIdle(device);
  cleanupSwapChain();

  defaultTexture.reset();
  textures.clear();

  vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);

  meshes.clear();

  for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
    vkDestroySemaphore(device, renderFinishedSemaphores[i], nullptr);
    vkDestroySemaphore(device, imageAvailableSemaphores[i], nullptr);
    vkDestroyFence(device, inFlightFences[i], nullptr);
  }
  vkDestroyCommandPool(device, commandPool, nullptr);

  vkDestroyDevice(device, nullptr);
  vkDestroySurfaceKHR(vk_inst, surface, nullptr);
  vkDestroyInstance(vk_inst, nullptr);

  glfwDestroyWindow(g_window);

  glfwTerminate();
}


void RD_Vulkan::ClearAll()
{
  m_diffColors.resize(0);
  m_diffTexId.resize(0);
}

HRDriverAllocInfo RD_Vulkan::AllocAll(HRDriverAllocInfo a_info)
{
  m_diffColors.resize(a_info.matNum * 3);
  m_diffTexId.resize(a_info.matNum);

  for (size_t i = 0; i < m_diffTexId.size(); i++)
    m_diffTexId[i] = -1;

  m_libPath = std::wstring(a_info.libraryPath);

  return a_info;
}

#pragma warning(disable:4996) // for wcscpy to be ok

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// typedef void (APIENTRYP PFNGLGENERATEMIPMAPPROC)(GLenum target);
// PFNGLGENERATEMIPMAPPROC glad_glGenerateMipmap;
//
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

bool RD_Vulkan::UpdateImage(int32_t a_texId, int32_t w, int32_t h, int32_t bpp, const void* a_data, pugi::xml_node a_texNode)
{
  if (a_data == nullptr) 
    return false; 

	uint8_t *convertedData = nullptr;
	
	if (bpp > 4) // well, perhaps this is not error, we just don't support hdr textures in this render
	{
		convertedData = new uint8_t[w*h*bpp/sizeof(float)];

		#pragma omp parallel for
		for (int y = 0; y < h; y++)
		{
			for (int x = 0; x < w; x++)
			{
				float r = ((float*)a_data)[(y*w + x) * 4 + 0];
				float g = ((float*)a_data)[(y*w + x) * 4 + 1];
				float b = ((float*)a_data)[(y*w + x) * 4 + 2];
				float a = ((float*)a_data)[(y*w + x) * 4 + 3];

				convertedData[(y*w + x) * 4 + 0] = uint8_t(clamp(r, 0.0, 1.0) * 255.0f);
				convertedData[(y*w + x) * 4 + 1] = uint8_t(clamp(g, 0.0, 1.0) * 255.0f);
				convertedData[(y*w + x) * 4 + 2] = uint8_t(clamp(b, 0.0, 1.0) * 255.0f);
				convertedData[(y*w + x) * 4 + 3] = uint8_t(clamp(a, 0.0, 1.0) * 255.0f);
			}
		}
	}

	if (bpp > 4) 
    delete [] convertedData;

  //textures.push_back(std::make_unique<Texture>(device));
  return true;
}


bool RD_Vulkan::UpdateMaterial(int32_t a_matId, pugi::xml_node a_materialNode)
{
  pugi::xml_node clrNode = a_materialNode.child(L"diffuse").child(L"color");
  pugi::xml_node texNode = a_materialNode.child(L"diffuse").child(L"texture");
  pugi::xml_node mtxNode = a_materialNode.child(L"diffuse").child(L"sampler").child(L"matrix");

  if (clrNode == nullptr)
    clrNode = a_materialNode.child(L"emission").child(L"color"); // no diffuse color ? => draw emission color instead!

  if (clrNode != nullptr)
  {
    const wchar_t* clrStr = nullptr;
    
    if (clrNode.attribute(L"val") != nullptr)
      clrStr = clrNode.attribute(L"val").as_string();
    else
      clrStr = clrNode.text().as_string();

    if (std::wstring(clrStr) != L"")
    {
      float color[3];
      std::wstringstream input(clrStr);
      input >> color[0] >> color[1] >> color[2];

      m_diffColors[a_matId * 3 + 0] = color[0];
      m_diffColors[a_matId * 3 + 1] = color[1];
      m_diffColors[a_matId * 3 + 2] = color[2];
    }
  }

  if (texNode != nullptr)
    m_diffTexId[a_matId] = texNode.attribute(L"id").as_int();
  else
    m_diffTexId[a_matId] = -1;

  return true;
}

bool RD_Vulkan::UpdateLight(int32_t a_lightIdId, pugi::xml_node a_lightNode)
{
  return true;
}


bool RD_Vulkan::UpdateCamera(pugi::xml_node a_camNode)
{
  if (a_camNode == nullptr)
    return true;

  this->camUseMatrices = false;

  if (std::wstring(a_camNode.attribute(L"type").as_string()) == L"two_matrices")
  {
    const wchar_t* m1 = a_camNode.child(L"mWorldView").text().as_string();
    const wchar_t* m2 = a_camNode.child(L"mProj").text().as_string();

    float mWorldView[16];
    float mProj[16];

    std::wstringstream str1(m1), str2(m2);
    for (int i = 0; i < 16; i++)
    {
      str1 >> mWorldView[i];
      str2 >> mProj[i];
    }

    this->camWorldViewMartrixTransposed = transpose(float4x4(mWorldView));
    this->camProjMatrixTransposed       = transpose(float4x4(mProj));
    this->camUseMatrices                = true;

    return true;
  }

  const wchar_t* camPosStr = a_camNode.child(L"position").text().as_string();
  const wchar_t* camLAtStr = a_camNode.child(L"look_at").text().as_string();
  const wchar_t* camUpStr  = a_camNode.child(L"up").text().as_string();
  //const wchar_t* testStr   = a_camNode.child(L"test").text().as_string();

  if (!a_camNode.child(L"fov").text().empty())
    camFov = a_camNode.child(L"fov").text().as_float();

  if (!a_camNode.child(L"nearClipPlane").text().empty())
    camNearPlane = a_camNode.child(L"nearClipPlane").text().as_float();

  if (!a_camNode.child(L"farClipPlane").text().empty())
    camFarPlane = a_camNode.child(L"farClipPlane").text().as_float();

  if (std::wstring(camPosStr) != L"")
  {
    std::wstringstream input(camPosStr);
    input >> camPos[0] >> camPos[1] >> camPos[2];
  }

  if (std::wstring(camLAtStr) != L"")
  {
    std::wstringstream input(camLAtStr);
    input >> camLookAt[0] >> camLookAt[1] >> camLookAt[2];
  }

  if (std::wstring(camUpStr) != L"")
  {
    std::wstringstream input(camUpStr);
    input >> camUp[0] >> camUp[1] >> camUp[2];
  }

  return true;
}

bool RD_Vulkan::UpdateSettings(pugi::xml_node a_settingsNode)
{
  if (a_settingsNode.child(L"width") != nullptr)
    m_width = a_settingsNode.child(L"width").text().as_int();

  if (a_settingsNode.child(L"height") != nullptr)
    m_height = a_settingsNode.child(L"height").text().as_int();

  if (m_width < 0 || m_height < 0)
  {
    if (m_pInfoCallBack != nullptr)
      m_pInfoCallBack(L"bad input resolution", L"RD_Vulkan::UpdateSettings", HR_SEVERITY_ERROR);
    return false;
  }

  return true;
}


bool RD_Vulkan::UpdateMesh(int32_t a_meshId, pugi::xml_node a_meshNode, const HRMeshDriverInput& a_input, const HRBatchInfo* a_batchList, int32_t a_listSize)
{
  if (a_input.triNum == 0) // don't support loading mesh from file 'a_fileName'
  {
    return true;
  }

  bool invalidMaterial = m_diffTexId.empty();

  std::vector<uint32_t> begins(1, 0);
  for (int i = 1; i < a_input.triNum; ++i) {
    if (a_input.triMatIndices[i] != a_input.triMatIndices[i - 1]) {
      begins.push_back(i);
    }
  }
  begins.push_back(a_input.triNum);

  for (int matId = 0; matId < begins.size() - 1; ++matId) {
    std::vector<uint32_t> indices;
    for (uint32_t i = begins[matId]; i < begins[matId + 1]; ++i) {
      indices.push_back(a_input.indices[i * 3]);
      indices.push_back(a_input.indices[i * 3 + 1]);
      indices.push_back(a_input.indices[i * 3 + 2]);
    }
    std::vector<Vertex> vertices;
    for (uint32_t i : indices) {
      float3 pos = { a_input.pos4f[4 * i], a_input.pos4f[4 * i + 1], a_input.pos4f[4 * i + 2] };
      float3 color = { a_input.norm4f[4 * i], a_input.norm4f[4 * i + 1], a_input.norm4f[4 * i + 2] };
      float2 tc = { a_input.texcoord2f[2 * i], a_input.texcoord2f[2 * i + 1] };
      Vertex vertex = { pos, color * 0.5f + 0.5f, tc };
      vertices.push_back(vertex);
    }

    std::unique_ptr<Mesh> mesh = std::make_unique<Mesh>(device);
    if (indices.size() >= 1 << 16) {
      std::vector<uint32_t> indices1(indices.size());
      for (int i = 0; i < indices.size(); ++i) {
        indices1[i] = i;
      }
      mesh->createMesh(vertices, indices1);
    } else {
      std::vector<uint16_t> indices1(indices.size());
      for (int i = 0; i < indices.size(); ++i) {
        indices1[i] = i;
      }
      mesh->createMesh(vertices, indices1);
    }
    meshes[a_meshId].addMesh(mesh);
  }
  createCommandBuffers();

  for (int32_t batchId = 0; batchId < a_listSize; batchId++)
  {
    HRBatchInfo batch = a_batchList[batchId];
    
    if (!invalidMaterial)
    {
      if (m_diffTexId[batch.matId] >= 0)
      {
        int texId = m_diffTexId[batch.matId];
      }
    }
    else
    {
    }
    const int drawElementsNum = batch.triEnd - batch.triBegin;
  }
  return true;
}


void RD_Vulkan::BeginScene(pugi::xml_node a_sceneNode)
{
  if (camUseMatrices)
  {
  }
  else
  {
  }
}

void RD_Vulkan::EndScene()
{

}

void RD_Vulkan::Draw()
{
  vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);
  vkResetFences(device, 1, &inFlightFences[currentFrame]);
  vkQueueWaitIdle(graphicsQueue);

  uint32_t imageIndex;
  VkResult result = vkAcquireNextImageKHR(device, swapchain, UINT64_MAX, imageAvailableSemaphores[currentFrame], VK_NULL_HANDLE, &imageIndex);

  if (result == VK_ERROR_OUT_OF_DATE_KHR) {
    recreateSwapChain();
    return;
  } else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
    throw std::runtime_error("failed to acquire swap chain image!");
  }

  updateUniformBuffer(imageIndex);

  VkSubmitInfo submitInfo = {};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submitInfo.waitSemaphoreCount = 1;
  submitInfo.pWaitSemaphores = &imageAvailableSemaphores[currentFrame];
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &commandBuffers[imageIndex];
  VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
  submitInfo.pWaitDstStageMask = waitStages;
  submitInfo.signalSemaphoreCount = 1;
  submitInfo.pSignalSemaphores = &renderFinishedSemaphores[currentFrame];
  vkResetFences(device, 1, &inFlightFences[currentFrame]);
  VK_CHECK_RESULT(vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFences[currentFrame]));
  VkPresentInfoKHR presentInfo = {};
  presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
  presentInfo.swapchainCount = 1;
  presentInfo.pSwapchains = &swapchain;
  presentInfo.pImageIndices = &imageIndex;
  presentInfo.waitSemaphoreCount = 1;
  presentInfo.pWaitSemaphores = &renderFinishedSemaphores[currentFrame];
  result = vkQueuePresentKHR(graphicsQueue, &presentInfo);

  if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) {
      recreateSwapChain();
  } else if (result != VK_SUCCESS) {
      throw std::runtime_error("failed to present swap chain image!");
  }

  currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
}

void RD_Vulkan::InstancesCollection::updateUniformBuffer(uint32_t current_image, const float4x4& view, const float4x4& proj) {
  std::vector<float4x4> resMatrices;
  for (auto& ubo : ubos) {
    ubo.setView(view);
    ubo.setProj(proj);
    resMatrices.push_back(ubo.getResultRef());
  }

  void* data;
  vkMapMemory(device, uniformBuffersMemory[current_image], 0, sizeof(resMatrices[0]) * resMatrices.size(), 0, &data);
  memcpy(data, resMatrices.data(), sizeof(resMatrices[0]) * resMatrices.size());
  vkUnmapMemory(device, uniformBuffersMemory[current_image]);
}

void RD_Vulkan::updateUniformBuffer(uint32_t current_image) {
  const float3 eye(camPos[0], camPos[1], camPos[2]);
  const float3 center(camLookAt[0], camLookAt[1], camLookAt[2]);
  const float3 up(camUp[0], camUp[1], camUp[2]);
  const float4x4 view = lookAtTransposed(eye, center, up);

  const float aspect = float(m_width) / float(m_height);
  float4x4 proj = projectionMatrixTransposed(camFov, aspect, camNearPlane, camFarPlane);
  proj.M(1, 1) *= -1.f;

  for (auto& instance : instances) {
    instance->updateUniformBuffer(current_image, view, proj);
  }
}

static inline void mat4x4_transpose(float M[16], const float N[16])
{
  for (int j = 0; j < 4; j++)
  {
    for (int i = 0; i < 4; i++)
      M[i * 4 + j] = N[j * 4 + i];
  }
}

bool RD_Vulkan::InstancesCollection::instancesUpdated(const std::vector<float4x4>& models) const {
  if (models.size() != ubos.size()) {
    return true;
  }
  for (int i = 0; i < models.size(); ++i) {
    for (int j = 0; j < 4; ++j) {
      if (models[i].row[j].x != ubos[i].getModel().row[j].x
        || models[i].row[j].y != ubos[i].getModel().row[j].y
        || models[i].row[j].z != ubos[i].getModel().row[j].z
        || models[i].row[j].w != ubos[i].getModel().row[j].w) {
        return true;
      }
    }
  }
  return false;
}

void RD_Vulkan::InstancesCollection::updateInstances(const std::vector<HydraLiteMath::float4x4>& models) {
  assert(models.size() < MAX_INSTANCES);
  ubos.resize(models.size());
  for (int i = 0; i < ubos.size(); ++i) {
    ubos[i].setModel(models[i]);
  }
}

void RD_Vulkan::InstanceMeshes(int32_t a_mesh_id, const float* a_matrices, int32_t a_instNum, const int* a_lightInstId, const int* a_remapId, const int* a_realInstId)
{
  int instancesId = 0;
  for (; instancesId < instances.size(); ++instancesId) {
    if (a_mesh_id == instances[instancesId]->getMeshId()) {
      break;
    }
  }
  std::vector<float4x4> models(a_instNum, a_matrices);
  for (int32_t i = 0; i < a_instNum; i++)
  {
    float matrixT2[16];
    mat4x4_transpose(matrixT2, (float*)(a_matrices + i * 16));
    models[i] = float4x4(matrixT2);
  }

  if (instancesId != instances.size() && instances[instancesId]->instancesUpdated(models)) {
    return;
  }

  if (instancesId == instances.size()) {
    instances.push_back(std::make_unique<InstancesCollection>(device, descriptorSetLayout, defaultTexture.get(), a_mesh_id, static_cast<int>(swapChainImages.size())));
    createCommandBuffers();
  }

  instances[instancesId]->updateInstances(models);
}


void RD_Vulkan::InstanceLights(int32_t a_light_id, const float* a_matrix, pugi::xml_node* a_custAttrArray, int32_t a_instNum, int32_t a_lightGroupId)
{

}

HRRenderUpdateInfo RD_Vulkan::HaveUpdateNow(int a_maxRaysPerPixel)
{
  HRRenderUpdateInfo res;
  res.finalUpdate   = true;
  res.haveUpdateFB  = true;
  res.progress      = 100.0f;
  return res;
}


void RD_Vulkan::GetFrameBufferHDR(int32_t w, int32_t h, float*   a_out, const wchar_t* a_layerName)
{

}

void RD_Vulkan::GetFrameBufferLDR(int32_t w, int32_t h, int32_t* a_out)
{
}
