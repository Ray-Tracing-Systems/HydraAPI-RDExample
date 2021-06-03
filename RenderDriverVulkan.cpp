// This is a personal academic project. Dear PVS-Studio, please check it.

// PVS-Studio Static Code Analyzer for C, C++, C#, and Java: http://www.viva64.com

#include <cassert>

#include <algorithm>
#include <array>
#include <chrono>
#include <set>
#include <ctime>


#include "RenderDriverVulkan.h"
#include "LiteMath.h"
#include "Bitmap.h"
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

extern bool screenShot;
extern bool screenshotAndExit;
extern bool reloadShaders;

struct Vertex {
  float3 pos;
  float3 normal;
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
    attributeDescriptions[0].offset = (uint32_t)offsetof(Vertex, pos);

    attributeDescriptions[1].binding = 0;
    attributeDescriptions[1].location = 1;
    attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
    attributeDescriptions[1].offset = (uint32_t)offsetof(Vertex, normal);

    attributeDescriptions[2].binding = 0;
    attributeDescriptions[2].location = 2;
    attributeDescriptions[2].format = VK_FORMAT_R32G32_SFLOAT;
    attributeDescriptions[2].offset = (uint32_t)offsetof(Vertex, texCoord);

    return attributeDescriptions;
  }
};

RD_Vulkan::QueueFamilyIndices RD_Vulkan::GetQueueFamilyIndex()
{
  QueueFamilyIndices familyIndices = {};
  uint32_t queueFamilyCount;

  vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, NULL);

  // Retrieve all queue families.
  std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
  vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, queueFamilies.data());

  // Now find a family that supports compute.
  size_t i = 0;
  for (; i < queueFamilies.size(); ++i)
  {
    VkQueueFamilyProperties props = queueFamilies[i];

    if (props.queueCount > 0 && (props.queueFlags & VK_QUEUE_GRAPHICS_BIT))
    {
      familyIndices.graphicsFamily = i;
    }

    VkBool32 presentSupport = false;
    vkGetPhysicalDeviceSurfaceSupportKHR(physicalDevice, i, surface, &presentSupport);
    if (props.queueCount > 0 && presentSupport) {
      familyIndices.presentFamily = i;
    }
  }

  return familyIndices;
}

void RD_Vulkan::createLogicalDevice()
{
  QueueFamilyIndices familyIndices = GetQueueFamilyIndex();

  std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
  std::set<uint32_t> uniqueQueueFamilies = { familyIndices.graphicsFamily, familyIndices.presentFamily };

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

  vkGetDeviceQueue(device, GetQueueFamilyIndex().graphicsFamily, 0, &graphicsQueue);
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

  assert(details.capabilities.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_SRC_BIT);

  return details;
}

void RD_Vulkan::createShadowMapRenderPass()
{
  VkAttachmentDescription depthAttachment = {};
  depthAttachment.format = findDepthFormat();
  depthAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
  depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
  depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;

  VkAttachmentReference depthAttachmentRef = {};
  depthAttachmentRef.attachment = 0;
  depthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

  VkSubpassDescription subpass = {};
  subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
  subpass.colorAttachmentCount = 0;
  subpass.pColorAttachments = nullptr;
  subpass.pDepthStencilAttachment = &depthAttachmentRef;

  VkSubpassDependency dependency = {};
  dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
  dependency.dstSubpass = 0;
  dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  dependency.srcAccessMask = 0;
  dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

  VkRenderPassCreateInfo renderPassInfo = {};
  renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
  renderPassInfo.attachmentCount = 1;
  renderPassInfo.pAttachments = &depthAttachment;
  renderPassInfo.subpassCount = 1;
  renderPassInfo.pSubpasses = &subpass;
  renderPassInfo.dependencyCount = 1;
  renderPassInfo.pDependencies = &dependency;

  VK_CHECK_RESULT(vkCreateRenderPass(device, &renderPassInfo, nullptr, &shadowMapRenderPass));
}

void RD_Vulkan::createGbufferRenderPass()
{
  VkAttachmentDescription colorAttachment = {};
  colorAttachment.format = swapChainImageFormat;
  colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;

  colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
  colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

  colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;

  colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  colorAttachment.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

  VkAttachmentDescription normalAttachment = {};
  normalAttachment.format = VK_FORMAT_R32G32B32A32_SFLOAT;
  normalAttachment.samples = VK_SAMPLE_COUNT_1_BIT;

  normalAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
  normalAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

  normalAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  normalAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;

  normalAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  normalAttachment.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

  VkAttachmentDescription depthAttachment = {};
  depthAttachment.format = findDepthFormat();
  depthAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
  depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
  depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;

  VkAttachmentReference colorAttachmentRef = {};
  colorAttachmentRef.attachment = 0;
  colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

  VkAttachmentReference normalAttachmentRef = {};
  normalAttachmentRef.attachment = 1;
  normalAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

  VkAttachmentReference depthAttachmentRef = {};
  depthAttachmentRef.attachment = 2;
  depthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

  std::array<VkAttachmentReference, 2> colorAttachments = { colorAttachmentRef, normalAttachmentRef };

  VkSubpassDescription subpass = {};
  subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
  subpass.colorAttachmentCount = colorAttachments.size();
  subpass.pColorAttachments = colorAttachments.data();
  subpass.pDepthStencilAttachment = &depthAttachmentRef;

  VkSubpassDependency dependency = {};
  dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
  dependency.dstSubpass = 0;
  dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  dependency.srcAccessMask = 0;
  dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

  std::array<VkAttachmentDescription, 3> attachments = { colorAttachment, normalAttachment, depthAttachment };
  VkRenderPassCreateInfo renderPassInfo = {};
  renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
  renderPassInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
  renderPassInfo.pAttachments = attachments.data();
  renderPassInfo.subpassCount = 1;
  renderPassInfo.pSubpasses = &subpass;
  renderPassInfo.dependencyCount = 1;
  renderPassInfo.pDependencies = &dependency;

  VK_CHECK_RESULT(vkCreateRenderPass(device, &renderPassInfo, nullptr, &gbufferRenderPass));
}

void RD_Vulkan::createResolveRenderPass()
{
  VkAttachmentDescription colorAttachment = {};
  colorAttachment.format = VK_FORMAT_R32G32B32A32_SFLOAT;
  colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;

  colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
  colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

  colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;

  colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  colorAttachment.finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;

  VkAttachmentReference colorAttachmentRef = {};
  colorAttachmentRef.attachment = 0;
  colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

  VkSubpassDescription subpass = {};
  subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
  subpass.colorAttachmentCount = 1;
  subpass.pColorAttachments = &colorAttachmentRef;
  subpass.pDepthStencilAttachment = nullptr;
  subpass.pResolveAttachments = nullptr;

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

  VK_CHECK_RESULT(vkCreateRenderPass(device, &renderPassInfo, nullptr, &resolveRenderPass));
}

void RD_Vulkan::createPostprocessRenderPass()
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
  subpass.pDepthStencilAttachment = nullptr;
  subpass.pResolveAttachments = nullptr;

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

  VK_CHECK_RESULT(vkCreateRenderPass(device, &renderPassInfo, nullptr, &postprocessRenderPass));
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

VkPipeline RD_Vulkan::createGraphicsPipeline(const PipelineConfig& config, VkPipelineLayout& layout) {
  std::vector<VkPipelineShaderStageCreateInfo> stages;

  auto vertShaderCode = readFile(config.vertexShaderPath);
  VkShaderModule vertShaderModule = createShaderModule(device, vertShaderCode);
  VkPipelineShaderStageCreateInfo vertPipelineShaderStageCreateInfo = {};
  vertPipelineShaderStageCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  vertPipelineShaderStageCreateInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
  vertPipelineShaderStageCreateInfo.module = vertShaderModule;
  vertPipelineShaderStageCreateInfo.pName = "main";

  stages.push_back(vertPipelineShaderStageCreateInfo);

  VkShaderModule fragShaderModule = VK_NULL_HANDLE;
  if (config.pixelShaderPath.has_value()) {
    std::vector<char> fragShaderCode = readFile(config.pixelShaderPath.value());

    fragShaderModule = createShaderModule(device, fragShaderCode);

    VkPipelineShaderStageCreateInfo fragPipelineShaderStageCreateInfo = {};
    fragPipelineShaderStageCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fragPipelineShaderStageCreateInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragPipelineShaderStageCreateInfo.module = fragShaderModule;
    fragPipelineShaderStageCreateInfo.pName = "main";

    stages.push_back(fragPipelineShaderStageCreateInfo);
  }

  VkPipelineVertexInputStateCreateInfo vertexInputStateCreateInfo = {};
  vertexInputStateCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
  auto bindingDescription = Vertex::getBindingDescription();
  auto attributeDescriptions = Vertex::getAttributeDescriptions();
  if (config.hasVertexBuffer) {
    vertexInputStateCreateInfo.vertexBindingDescriptionCount = 1;
    vertexInputStateCreateInfo.pVertexBindingDescriptions = &bindingDescription;
    vertexInputStateCreateInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
    vertexInputStateCreateInfo.pVertexAttributeDescriptions = attributeDescriptions.data();
  } else {
    vertexInputStateCreateInfo.vertexBindingDescriptionCount = 0;
  }

  VkPipelineInputAssemblyStateCreateInfo inputAssemblyDesc = {};
  inputAssemblyDesc.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
  inputAssemblyDesc.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
  inputAssemblyDesc.primitiveRestartEnable = VK_FALSE;

  VkViewport viewport = {};
  viewport.x = 0;
  viewport.y = 0;
  viewport.width = config.width;
  viewport.height = config.height;
  viewport.minDepth = 0;
  viewport.maxDepth = 1;

  VkRect2D scissor = {};
  scissor.offset.x = scissor.offset.y = 0;
  scissor.extent.width = config.width;
  scissor.extent.height = config.height;

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
  rasterizationState.cullMode = config.cullingBits;
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

  std::vector<VkPipelineColorBlendAttachmentState> blendStates(config.rtCount, blendStateAttach);

  VkPipelineColorBlendStateCreateInfo colorBlending = {};
  colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
  colorBlending.logicOpEnable = VK_FALSE;
  colorBlending.logicOp = VK_LOGIC_OP_COPY; // Optional
  colorBlending.attachmentCount = blendStates.size();
  colorBlending.pAttachments = blendStates.data();
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
  pipelineLayoutCreateInfo.pSetLayouts = &config.descriptorSetLayout;
  pipelineLayoutCreateInfo.pushConstantRangeCount = config.pushConstants.size();
  pipelineLayoutCreateInfo.pPushConstantRanges = config.pushConstants.data();

  VkPipelineDepthStencilStateCreateInfo depthStencil = {};
  depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
  depthStencil.depthTestEnable = VK_TRUE;
  depthStencil.depthWriteEnable = VK_TRUE;
  depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
  depthStencil.depthBoundsTestEnable = VK_FALSE;
  depthStencil.minDepthBounds = 0.0f; // Optional
  depthStencil.maxDepthBounds = 1.0f; // Optional
  depthStencil.stencilTestEnable = VK_FALSE;
  depthStencil.front = {}; // Optional
  depthStencil.back = {}; // Optional

  VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, nullptr, &layout));

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
  graphicsPipelineCreateInfo.layout = layout;
  graphicsPipelineCreateInfo.renderPass = config.renderPass;
  graphicsPipelineCreateInfo.subpass = 0;
  graphicsPipelineCreateInfo.basePipelineHandle = VK_NULL_HANDLE;
  graphicsPipelineCreateInfo.basePipelineIndex = -1; // Optional
  graphicsPipelineCreateInfo.pDepthStencilState = &depthStencil;

  VkPipeline pipeline = {};
  VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &graphicsPipelineCreateInfo, nullptr, &pipeline));

  if (fragShaderModule != VK_NULL_HANDLE) {
    vkDestroyShaderModule(device, fragShaderModule, nullptr);
  }
  vkDestroyShaderModule(device, vertShaderModule, nullptr);
  return pipeline;
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

void RD_Vulkan::createVertexBuffer() {
  VkDeviceSize bufferSize = sizeof(vertices[0]) * vertices.size();

  VkBuffer stagingBuffer;
  VkDeviceMemory stagingBufferMemory;
  BufferManager::get().createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

  void* data;
  vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
  memcpy(data, vertices.data(), (size_t)bufferSize);
  vkUnmapMemory(device, stagingBufferMemory);

  BufferManager::get().createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, globalVertexBuffer, globalVertexBufferMemory);

  BufferManager::get().copyBuffer(stagingBuffer, globalVertexBuffer, bufferSize);
  vkDestroyBuffer(device, stagingBuffer, nullptr);
  vkFreeMemory(device, stagingBufferMemory, nullptr);
}

void RD_Vulkan::createIndexBuffer() {
  VkDeviceSize bufferSize = sizeof(indices[0]) * indices.size();

  VkBuffer stagingBuffer;
  VkDeviceMemory stagingBufferMemory;
  BufferManager::get().createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

  void* data;
  vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
  memcpy(data, indices.data(), (size_t)bufferSize);
  vkUnmapMemory(device, stagingBufferMemory);

  BufferManager::get().createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, globalIndexBuffer, globalIndexBufferMemory);

  BufferManager::get().copyBuffer(stagingBuffer, globalIndexBuffer, bufferSize);

  vkDestroyBuffer(device, stagingBuffer, nullptr);
  vkFreeMemory(device, stagingBufferMemory, nullptr);
}

void RD_Vulkan::BufferManager::copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size) {
  SingleTimeCommandsContext context;

  VkBufferCopy copyRegion = {};
  copyRegion.size = size;
  vkCmdCopyBuffer(context.getCB(), srcBuffer, dstBuffer, 1, &copyRegion);
}

bool hasStencilComponent(VkFormat format) {
  return format == VK_FORMAT_D32_SFLOAT_S8_UINT || format == VK_FORMAT_D24_UNORM_S8_UINT;
}

static VkImageAspectFlags get_aspect_bits(VkImageLayout target_layout, VkFormat format) {
  if (target_layout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL || target_layout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL) {
    VkImageAspectFlags aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;

    if (hasStencilComponent(format)) {
      aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
    }
    return aspectMask;
  }
  else {
    return VK_IMAGE_ASPECT_COLOR_BIT;
  }
}

void RD_Vulkan::BufferManager::transitionImageLayout(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout, uint32_t mipLevels) {
  SingleTimeCommandsContext context;

  VkImageMemoryBarrier barrier = {};
  barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  barrier.oldLayout = oldLayout;
  barrier.newLayout = newLayout;
  barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.image = image;
  barrier.subresourceRange.aspectMask = get_aspect_bits(newLayout, format);
  barrier.subresourceRange.baseMipLevel = 0;
  barrier.subresourceRange.levelCount = mipLevels;
  barrier.subresourceRange.baseArrayLayer = 0;
  barrier.subresourceRange.layerCount = 1;
  barrier.srcAccessMask = 0; // TODO
  barrier.dstAccessMask = 0; // TODO

  VkPipelineStageFlags sourceStage = {};
  VkPipelineStageFlags destinationStage = {};

  if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

    sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
  } else if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL) {
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

    sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
  } else if (newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
  } else if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && (newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL || newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL)) {
    barrier.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

    sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    destinationStage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
  } else if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL) {
    barrier.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    destinationStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  } else {
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

void RD_Vulkan::BufferManager::copyImageToBuffer(VkImage image, VkBuffer buffer, uint32_t width, uint32_t height) {
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

  vkCmdCopyImageToBuffer(
    context.getCB(),
    image,
    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
    buffer,
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
  createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;

  QueueFamilyIndices familyIndices = GetQueueFamilyIndex();
  uint32_t queueFamilyIndices[] = { familyIndices.graphicsFamily, familyIndices.presentFamily };

  if (familyIndices.graphicsFamily != familyIndices.presentFamily) {
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
    swapChainImageViews[i] = BufferManager::get().createImageView(swapChainImages[i], swapChainImageFormat, VK_IMAGE_ASPECT_COLOR_BIT, 1);
  }
}

void RD_Vulkan::createFramebuffers() {
  {
    VkFramebufferCreateInfo framebufferCreateInfo = {};
    framebufferCreateInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    framebufferCreateInfo.attachmentCount = 1;
    framebufferCreateInfo.pAttachments = &shadowMapImageView;
    framebufferCreateInfo.width = SHADOW_MAP_RESOLUTION;
    framebufferCreateInfo.height = SHADOW_MAP_RESOLUTION;
    framebufferCreateInfo.layers = 1;
    framebufferCreateInfo.renderPass = shadowMapRenderPass;
    VK_CHECK_RESULT(vkCreateFramebuffer(device, &framebufferCreateInfo, nullptr, &shadowMapFramebuffer));
  }

  std::array<VkImageView, 3> attachments = {
      colorImageView,
      normalImageView,
      depthImageView,
  };

  VkFramebufferCreateInfo framebufferCreateInfo = {};
  framebufferCreateInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
  framebufferCreateInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
  framebufferCreateInfo.pAttachments = attachments.data();
  framebufferCreateInfo.width = swapChainExtent.width;
  framebufferCreateInfo.height = swapChainExtent.height;
  framebufferCreateInfo.layers = 1;
  framebufferCreateInfo.renderPass = gbufferRenderPass;
  VK_CHECK_RESULT(vkCreateFramebuffer(device, &framebufferCreateInfo, nullptr, &gbufferFramebuffer));

  {
    VkFramebufferCreateInfo framebufferCreateInfo = {};
    framebufferCreateInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    framebufferCreateInfo.attachmentCount = 1;
    framebufferCreateInfo.pAttachments = &frameImageView;
    framebufferCreateInfo.width = swapChainExtent.width;
    framebufferCreateInfo.height = swapChainExtent.height;
    framebufferCreateInfo.layers = 1;
    framebufferCreateInfo.renderPass = resolveRenderPass;
    VK_CHECK_RESULT(vkCreateFramebuffer(device, &framebufferCreateInfo, nullptr, &resolveFramebuffer));
  }

  swapChainFramebuffers.resize(swapChainImageViews.size());
  for (int i = 0; i < swapChainImages.size(); ++i)
  {
    std::array<VkImageView, 1> attachments = {
      swapChainImageViews[i]
    };

    VkFramebufferCreateInfo framebufferCreateInfo = {};
    framebufferCreateInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    framebufferCreateInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
    framebufferCreateInfo.pAttachments = attachments.data();
    framebufferCreateInfo.width = swapChainExtent.width;
    framebufferCreateInfo.height = swapChainExtent.height;
    framebufferCreateInfo.layers = 1;
    framebufferCreateInfo.renderPass = postprocessRenderPass;
    VK_CHECK_RESULT(vkCreateFramebuffer(device, &framebufferCreateInfo, nullptr, &swapChainFramebuffers[i]));
  }
}

void RD_Vulkan::createCommandPool() {
  VkCommandPoolCreateInfo commandPoolCreateInfo = {};
  commandPoolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  commandPoolCreateInfo.queueFamilyIndex = GetQueueFamilyIndex().graphicsFamily;
  commandPoolCreateInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  VK_CHECK_RESULT(vkCreateCommandPool(device, &commandPoolCreateInfo, nullptr, &commandPool));
}

void RD_Vulkan::BufferManager::createImage(uint32_t width, uint32_t height, uint32_t mip_levels,
  VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage,
  VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& imageMemory)
{
  VkImageCreateInfo imageInfo = {};
  imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  imageInfo.imageType = VK_IMAGE_TYPE_2D;
  imageInfo.extent.width = width;
  imageInfo.extent.height = height;
  imageInfo.extent.depth = 1;
  imageInfo.mipLevels = mip_levels;
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

template<typename T>
void RD_Vulkan::Texture::createTextureImage(uint32_t width, uint32_t height, const T* image, VkFormat format) {
  VkDeviceSize imageSize = width * height * 4 * sizeof(T);
  mipLevels = static_cast<uint32_t>(std::floor(std::log2(std::max(width, height)))) + 1;

  VkBuffer stagingBuffer;
  VkDeviceMemory stagingBufferMemory;

  BufferManager::get().createBuffer(imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

  void* data;
  vkMapMemory(deviceRef, stagingBufferMemory, 0, imageSize, 0, &data);
  memcpy(data, image, static_cast<size_t>(imageSize));
  vkUnmapMemory(deviceRef, stagingBufferMemory);

  BufferManager::get().createImage(width, height, mipLevels, format, VK_IMAGE_TILING_OPTIMAL,
    VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, textureImage, textureImageMemory);

  BufferManager::get().transitionImageLayout(textureImage, format, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, mipLevels);
  BufferManager::get().copyBufferToImage(stagingBuffer, textureImage, width, height);
  BufferManager::get().generateMipmaps(textureImage, format, width, height, mipLevels);

  vkDestroyBuffer(deviceRef, stagingBuffer, nullptr);
  vkFreeMemory(deviceRef, stagingBufferMemory, nullptr);
}

static void generate_mipmaps(VkPhysicalDevice device, VkCommandBuffer comBuf, VkImage image, VkFormat imageFormat, int32_t texWidth, int32_t texHeight, uint32_t mipLevels) {
  VkFormatProperties formatProperties;
  vkGetPhysicalDeviceFormatProperties(device, imageFormat, &formatProperties);

  if (!(formatProperties.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT)) {
    throw std::runtime_error("texture image format does not support linear blitting!");
  }

  VkImageMemoryBarrier barrier = {};
  barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  barrier.image = image;
  barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  barrier.subresourceRange.baseArrayLayer = 0;
  barrier.subresourceRange.layerCount = 1;
  barrier.subresourceRange.levelCount = 1;

  int32_t mipWidth = texWidth;
  int32_t mipHeight = texHeight;

  for (uint32_t i = 1; i < mipLevels; i++) {
    barrier.subresourceRange.baseMipLevel = i - 1;
    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

    vkCmdPipelineBarrier(comBuf,
      VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
      0, nullptr,
      0, nullptr,
      1, &barrier);

    VkImageBlit blit = {};
    blit.srcOffsets[0] = { 0, 0, 0 };
    blit.srcOffsets[1] = { mipWidth, mipHeight, 1 };
    blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    blit.srcSubresource.mipLevel = i - 1;
    blit.srcSubresource.baseArrayLayer = 0;
    blit.srcSubresource.layerCount = 1;
    blit.dstOffsets[0] = { 0, 0, 0 };
    blit.dstOffsets[1] = { mipWidth > 1 ? mipWidth / 2 : 1, mipHeight > 1 ? mipHeight / 2 : 1, 1 };
    blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    blit.dstSubresource.mipLevel = i;
    blit.dstSubresource.baseArrayLayer = 0;
    blit.dstSubresource.layerCount = 1;

    vkCmdBlitImage(comBuf,
      image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
      image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
      1, &blit,
      VK_FILTER_LINEAR);

    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    vkCmdPipelineBarrier(comBuf,
      VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
      0, nullptr,
      0, nullptr,
      1, &barrier);

    if (mipWidth > 1) mipWidth /= 2;
    if (mipHeight > 1) mipHeight /= 2;
  }

  barrier.subresourceRange.baseMipLevel = mipLevels - 1;
  barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

  vkCmdPipelineBarrier(comBuf,
    VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
    0, nullptr,
    0, nullptr,
    1, &barrier);
}

void RD_Vulkan::BufferManager::generateMipmaps(VkImage image, VkFormat imageFormat, int32_t texWidth, int32_t texHeight, uint32_t mipLevels) {
  SingleTimeCommandsContext ctx;
  generate_mipmaps(physicalDeviceRef, ctx.getCB(), image, imageFormat, texWidth, texHeight, mipLevels);
}

static void copy_image(VkCommandBuffer comBuf, VkImage src, VkImage dst, uint32_t width, uint32_t height) {
  VkImageCopy copyRegion = {};
  copyRegion.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  copyRegion.srcSubresource.layerCount = 1;
  copyRegion.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  copyRegion.dstSubresource.layerCount = 1;
  copyRegion.extent.width = width;
  copyRegion.extent.height = height;
  copyRegion.extent.depth = 1;

  vkCmdCopyImage(comBuf, src, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copyRegion);
}

VkImageView RD_Vulkan::BufferManager::createImageView(VkImage image, VkFormat format, VkImageAspectFlags aspectFlags, uint32_t mipLevels) {
  VkImageViewCreateInfo viewInfo = {};
  viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  viewInfo.image = image;
  viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
  viewInfo.format = format;
  viewInfo.subresourceRange.aspectMask = aspectFlags;
  viewInfo.subresourceRange.baseMipLevel = 0;
  viewInfo.subresourceRange.levelCount = mipLevels;
  viewInfo.subresourceRange.baseArrayLayer = 0;
  viewInfo.subresourceRange.layerCount = 1;

  VkImageView imageView;
  VK_CHECK_RESULT(vkCreateImageView(deviceRef, &viewInfo, nullptr, &imageView));

  return imageView;
}

void RD_Vulkan::Texture::createTextureImageView(VkFormat format) {
  textureImageView = BufferManager::get().createImageView(textureImage, format, VK_IMAGE_ASPECT_COLOR_BIT, mipLevels);
}

void RD_Vulkan::createCommandBuffers() {
  commandBuffers.resize(swapChainImages.size());

  VkCommandBufferAllocateInfo allocInfo = {};
  allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  allocInfo.commandPool = commandPool;
  allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  allocInfo.commandBufferCount = static_cast<uint32_t>(commandBuffers.size());

  VK_CHECK_RESULT(vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data()));
}

void RD_Vulkan::prepareCommandBuffers(uint32_t current_image) {
  VkCommandBufferBeginInfo beginInfo = {};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  beginInfo.flags = 0;
  beginInfo.pInheritanceInfo = nullptr;
  VK_CHECK_RESULT(vkBeginCommandBuffer(commandBuffers[current_image], &beginInfo));

  VkDeviceSize zeroOffset = 0;
  if (hasDirectLight)
  {
    VkRenderPassBeginInfo shadowMapRenderPassBeginInfo = {};
    shadowMapRenderPassBeginInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    shadowMapRenderPassBeginInfo.renderPass = shadowMapRenderPass;
    shadowMapRenderPassBeginInfo.framebuffer = shadowMapFramebuffer;
    shadowMapRenderPassBeginInfo.renderArea.offset = { 0, 0 };
    shadowMapRenderPassBeginInfo.renderArea.extent = VkExtent2D{ SHADOW_MAP_RESOLUTION, SHADOW_MAP_RESOLUTION };
    VkClearValue shadowMapClearValues = {};
    shadowMapClearValues.depthStencil = { 1.0f, 0 };
    shadowMapRenderPassBeginInfo.clearValueCount = 1;
    shadowMapRenderPassBeginInfo.pClearValues = &shadowMapClearValues;
    vkCmdBeginRenderPass(commandBuffers[current_image], &shadowMapRenderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

    vkCmdBindPipeline(commandBuffers[current_image], VK_PIPELINE_BIND_POINT_GRAPHICS, shadowMapPipeline);

    vkCmdBindVertexBuffers(commandBuffers[current_image], 0, 1, &globalVertexBuffer, &zeroOffset);
    vkCmdBindIndexBuffer(commandBuffers[current_image], globalIndexBuffer, 0, VK_INDEX_TYPE_UINT32);
    vkCmdPushConstants(commandBuffers[current_image], shadowMapPipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(float4x4), &lighttm);
    for (auto& modelInstance : modelInstances) {
      for (auto& subMeshes : modelInstance.parts) {
        vkCmdPushConstants(commandBuffers[current_image], shadowMapPipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, sizeof(float4x4), sizeof(float4), &materials[subMeshes.mesh.materialId].color);
        vkCmdBindDescriptorSets(commandBuffers[current_image], VK_PIPELINE_BIND_POINT_GRAPHICS, shadowMapPipelineLayout, 0, 1, &materialsShadowDs, 0, nullptr);
        vkCmdDrawIndexed(commandBuffers[current_image], subMeshes.mesh.incidesCount, subMeshes.matricesCount, subMeshes.mesh.indicesOffset, 0, subMeshes.matricesOffset);
      }
    }

    vkCmdEndRenderPass(commandBuffers[current_image]);
  }

  VkRenderPassBeginInfo gbufferRenderPassBeginInfo = {};
  gbufferRenderPassBeginInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
  gbufferRenderPassBeginInfo.renderPass = gbufferRenderPass;
  gbufferRenderPassBeginInfo.framebuffer = gbufferFramebuffer;
  gbufferRenderPassBeginInfo.renderArea.offset = { 0, 0 };
  gbufferRenderPassBeginInfo.renderArea.extent = swapChainExtent;
  std::array<VkClearValue, 3> gbufferClearValues = {};
  gbufferClearValues[0].color = { 0.0f, 0.0f, 0.0f, 1.0f };
  gbufferClearValues[1].color = { 0.0f, 0.0f, 0.0f, 1.0f };
  gbufferClearValues[2].depthStencil = { 1.0f, 0 };
  gbufferRenderPassBeginInfo.clearValueCount = static_cast<uint32_t>(gbufferClearValues.size());
  gbufferRenderPassBeginInfo.pClearValues = gbufferClearValues.data();
  vkCmdBeginRenderPass(commandBuffers[current_image], &gbufferRenderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

  vkCmdBindPipeline(commandBuffers[current_image], VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);

  vkCmdBindVertexBuffers(commandBuffers[current_image], 0, 1, &globalVertexBuffer, &zeroOffset);
  vkCmdBindIndexBuffer(commandBuffers[current_image], globalIndexBuffer, 0, VK_INDEX_TYPE_UINT32);
  vkCmdPushConstants(commandBuffers[current_image], gbufferPipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(float4x4), &globtm);
  for (auto& modelInstance : modelInstances) {
    for (auto& subMeshes : modelInstance.parts) {
      vkCmdPushConstants(commandBuffers[current_image], gbufferPipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, sizeof(float4x4), sizeof(float4), &materials[subMeshes.mesh.materialId].color);
      vkCmdBindDescriptorSets(commandBuffers[current_image], VK_PIPELINE_BIND_POINT_GRAPHICS, gbufferPipelineLayout, 0, 1, &materialsLib[subMeshes.mesh.materialId], 0, nullptr);
      vkCmdDrawIndexed(commandBuffers[current_image], subMeshes.mesh.incidesCount, subMeshes.matricesCount, subMeshes.mesh.indicesOffset, 0, subMeshes.matricesOffset);
    }
  }

  vkCmdEndRenderPass(commandBuffers[current_image]);

  VkRenderPassBeginInfo renderPassBeginInfo = {};
  renderPassBeginInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
  renderPassBeginInfo.renderPass = resolveRenderPass;
  renderPassBeginInfo.framebuffer = resolveFramebuffer;
  renderPassBeginInfo.renderArea.offset = { 0, 0 };
  renderPassBeginInfo.renderArea.extent = swapChainExtent;
  std::array<VkClearValue, 1> clearValues = {};
  clearValues[0].color = { 0.0f, 0.0f, 0.0f, 1.0f };
  renderPassBeginInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
  renderPassBeginInfo.pClearValues = clearValues.data();
  vkCmdBeginRenderPass(commandBuffers[current_image], &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

  vkCmdBindPipeline(commandBuffers[current_image], VK_PIPELINE_BIND_POINT_GRAPHICS, resolvePipeline);
  vkCmdPushConstants(commandBuffers[current_image], resolvePipelineLayout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(float4x4), &lighttm);
  vkCmdBindDescriptorSets(commandBuffers[current_image], VK_PIPELINE_BIND_POINT_GRAPHICS, resolvePipelineLayout, 0, 1, &resolveDescriptorSets, 0, nullptr);
  vkCmdDraw(commandBuffers[current_image], 3, 1, 0, 0);

  vkCmdEndRenderPass(commandBuffers[current_image]);

  VkImageMemoryBarrier dstBarrier = {};
  dstBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  dstBarrier.image = frameMipchainImage;
  dstBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  dstBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  dstBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  dstBarrier.subresourceRange.baseArrayLayer = 0;
  dstBarrier.subresourceRange.layerCount = 1;
  dstBarrier.subresourceRange.levelCount = screenMipLevels;
  dstBarrier.subresourceRange.baseMipLevel = 0;
  dstBarrier.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  dstBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  dstBarrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
  dstBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

  vkCmdPipelineBarrier(commandBuffers[current_image],
    VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
    0, nullptr,
    0, nullptr,
    1, &dstBarrier);
  copy_image(commandBuffers[current_image], frameImage, frameMipchainImage, swapChainExtent.width, swapChainExtent.height);
  generate_mipmaps(physicalDevice, commandBuffers[current_image], frameMipchainImage, VK_FORMAT_R32G32B32A32_SFLOAT, swapChainExtent.width, swapChainExtent.height, screenMipLevels);

  renderPassBeginInfo.renderPass = postprocessRenderPass;
  renderPassBeginInfo.framebuffer = swapChainFramebuffers[current_image];
  vkCmdBeginRenderPass(commandBuffers[current_image], &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

  vkCmdBindPipeline(commandBuffers[current_image], VK_PIPELINE_BIND_POINT_GRAPHICS, postprocessPipeline);
  vkCmdBindDescriptorSets(commandBuffers[current_image], VK_PIPELINE_BIND_POINT_GRAPHICS, postprocessPipelineLayout, 0, 1, &postprocessDescriptorSets[current_image], 0, nullptr);
  vkCmdDraw(commandBuffers[current_image], 3, 1, 0, 0);

  vkCmdEndRenderPass(commandBuffers[current_image]);

  if (screenshotState == ScreenshotState::REQUIRED) {
    VkDeviceSize imageSize = m_width * m_height * 4;

    BufferManager::get().createBuffer(imageSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, screenshotBuffer, screenshotBufferMemory);

    VkImageMemoryBarrier dstBarrier = {};
    dstBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    dstBarrier.image = swapChainImages[current_image];
    dstBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    dstBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    dstBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    dstBarrier.subresourceRange.baseArrayLayer = 0;
    dstBarrier.subresourceRange.layerCount = 1;
    dstBarrier.subresourceRange.levelCount = 1;
    dstBarrier.subresourceRange.baseMipLevel = 0;
    dstBarrier.oldLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    dstBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    dstBarrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    dstBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

    vkCmdPipelineBarrier(commandBuffers[current_image],
      VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
      0, nullptr,
      0, nullptr,
      1, &dstBarrier);

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
        static_cast<uint32_t>(m_width),
        static_cast<uint32_t>(m_height),
        1
    };

    vkCmdCopyImageToBuffer(
      commandBuffers[current_image],
      swapChainImages[current_image],
      VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
      screenshotBuffer,
      1,
      &region
    );

    screenshotState = ScreenshotState::IN_PROGRESS;
    screenshotFrameIdx = current_image;

    {
      VkImageMemoryBarrier dstBarrier = {};
      dstBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
      dstBarrier.image = swapChainImages[current_image];
      dstBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      dstBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      dstBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
      dstBarrier.subresourceRange.baseArrayLayer = 0;
      dstBarrier.subresourceRange.layerCount = 1;
      dstBarrier.subresourceRange.levelCount = 1;
      dstBarrier.subresourceRange.baseMipLevel = 0;
      dstBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
      dstBarrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
      dstBarrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
      dstBarrier.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

      vkCmdPipelineBarrier(commandBuffers[current_image],
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0,
        0, nullptr,
        0, nullptr,
        1, &dstBarrier);
    }
  }

  VK_CHECK_RESULT(vkEndCommandBuffer(commandBuffers[current_image]));
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

void RD_Vulkan::cleanupSwapChain() {
  vkDestroyImageView(device, colorImageView, nullptr);
  vkDestroyImage(device, colorImage, nullptr);
  vkFreeMemory(device, colorImageMemory, nullptr);

  vkDestroyImageView(device, normalImageView, nullptr);
  vkDestroyImage(device, normalImage, nullptr);
  vkFreeMemory(device, normalImageMemory, nullptr);

  vkDestroyImageView(device, depthImageView, nullptr);
  vkDestroyImage(device, depthImage, nullptr);
  vkFreeMemory(device, depthImageMemory, nullptr);

  vkDestroyImageView(device, shadowMapImageView, nullptr);
  vkDestroyImage(device, shadowMapImage, nullptr);
  vkFreeMemory(device, shadowMapImageMemory, nullptr);

  vkDestroyImageView(device, frameImageView, nullptr);
  vkDestroyImage(device, frameImage, nullptr);
  vkFreeMemory(device, frameImageMemory, nullptr);

  vkDestroyImageView(device, frameMipchainImageView, nullptr);
  vkDestroyImage(device, frameMipchainImage, nullptr);
  vkFreeMemory(device, frameMipchainImageMemory, nullptr);

  vkDestroySampler(device, colorImageSampler, nullptr);
  vkDestroySampler(device, normalImageSampler, nullptr);
  vkDestroySampler(device, depthImageSampler, nullptr);
  vkDestroySampler(device, shadowMapImageSampler, nullptr);
  vkDestroySampler(device, frameImageSampler, nullptr);
  vkDestroySampler(device, frameMipchainImageSampler, nullptr);

  vkDestroyBuffer(device, resolveConstants, nullptr);
  vkFreeMemory(device, resolveConstantsMemory, nullptr);

  vkDestroyBuffer(device, lightsBuffer, nullptr);
  vkFreeMemory(device, lightsBufferMemory, nullptr);

  vkDestroyDescriptorPool(device, shadowMapDescriptorPool, nullptr);
  vkDestroyDescriptorPool(device, gbufferDescriptorPool, nullptr);
  vkDestroyDescriptorPool(device, resolveDescriptorPool, nullptr);
  vkDestroyDescriptorPool(device, postprocessDescriptorPool, nullptr);

  vkDestroyFramebuffer(device, shadowMapFramebuffer, nullptr);
  vkDestroyFramebuffer(device, gbufferFramebuffer, nullptr);
  vkDestroyFramebuffer(device, resolveFramebuffer, nullptr);

  for (auto framebuffer : swapChainFramebuffers) {
    vkDestroyFramebuffer(device, framebuffer, nullptr);
  }
  vkFreeCommandBuffers(device, commandPool, static_cast<uint32_t>(commandBuffers.size()), commandBuffers.data());

  destroyPipelines();
  vkDestroyRenderPass(device, shadowMapRenderPass, nullptr);
  vkDestroyRenderPass(device, gbufferRenderPass, nullptr);
  vkDestroyRenderPass(device, resolveRenderPass, nullptr);
  vkDestroyRenderPass(device, postprocessRenderPass, nullptr);

  for (auto imageView : swapChainImageViews) {
    vkDestroyImageView(device, imageView, nullptr);
  }

  vkDestroySwapchainKHR(device, swapchain, nullptr);
}

void RD_Vulkan::createPipelines() {
  PipelineConfig geometryConfig;
  geometryConfig.vertexShaderPath = "shaders/vert.spv";
  geometryConfig.pixelShaderPath = "shaders/frag.spv";
  geometryConfig.hasVertexBuffer = true;
  geometryConfig.renderPass = gbufferRenderPass;
  geometryConfig.descriptorSetLayout = gbufferDescriptorSetLayout;
  geometryConfig.rtCount = 2;
  geometryConfig.pushConstants.emplace_back(VkPushConstantRange{ VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(float4x4) + sizeof(float4) });
  geometryConfig.width = swapChainExtent.width;
  geometryConfig.height = swapChainExtent.height;
  graphicsPipeline = createGraphicsPipeline(geometryConfig, gbufferPipelineLayout);

  PipelineConfig shadowMapConfig;
  shadowMapConfig.vertexShaderPath = "shaders/vert.spv";
  shadowMapConfig.hasVertexBuffer = true;
  shadowMapConfig.renderPass = shadowMapRenderPass;
  shadowMapConfig.descriptorSetLayout = shadowMapDescriptorSetLayout;
  shadowMapConfig.rtCount = 0;
  shadowMapConfig.pushConstants.emplace_back(VkPushConstantRange{ VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(float4x4) + sizeof(float4) });
  shadowMapConfig.width = SHADOW_MAP_RESOLUTION;
  shadowMapConfig.height = SHADOW_MAP_RESOLUTION;
  shadowMapConfig.cullingBits = VK_CULL_MODE_BACK_BIT;
  shadowMapPipeline = createGraphicsPipeline(shadowMapConfig, shadowMapPipelineLayout);

  PipelineConfig resolveConfig;
  resolveConfig.vertexShaderPath = "shaders/full_screen.spv";
  resolveConfig.pixelShaderPath = "shaders/resolve.spv";
  resolveConfig.renderPass = resolveRenderPass;
  resolveConfig.descriptorSetLayout = resolveDescriptorSetLayout;
  resolveConfig.width = swapChainExtent.width;
  resolveConfig.height = swapChainExtent.height;
  resolveConfig.pushConstants.emplace_back(VkPushConstantRange{ VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(float4x4) });
  resolvePipeline = createGraphicsPipeline(resolveConfig, resolvePipelineLayout);

  PipelineConfig postprocessConfig;
  postprocessConfig.vertexShaderPath = "shaders/full_screen.spv";
  postprocessConfig.pixelShaderPath = "shaders/postprocess.spv";
  postprocessConfig.renderPass = postprocessRenderPass;
  postprocessConfig.descriptorSetLayout = postprocessDescriptorSetLayout;
  postprocessConfig.width = swapChainExtent.width;
  postprocessConfig.height = swapChainExtent.height;
  postprocessPipeline = createGraphicsPipeline(postprocessConfig, postprocessPipelineLayout);
}

void RD_Vulkan::recreateSwapChain() {
  vkDeviceWaitIdle(device);

  cleanupSwapChain();

  createSwapChain();
  createImageViews();
  createShadowMapRenderPass();
  createGbufferRenderPass();
  createResolveRenderPass();
  createPostprocessRenderPass();
  createPipelines();
  createColorResources();
  createDepthResources();
  createFramebuffers();
}

static VkDescriptorSetLayout create_descriptors_set_layout(VkDevice device, const uint32_t uniform_buffers_count, const uint32_t textures_count, const VkShaderStageFlagBits buffers_bits) {
  std::vector<VkDescriptorSetLayoutBinding> bindings(uniform_buffers_count + textures_count);
  for (uint32_t i = 0; i < uniform_buffers_count; ++i) {
    VkDescriptorSetLayoutBinding uboLayoutBinding = {};
    bindings[i].binding = i;
    bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    bindings[i].descriptorCount = 1;
    bindings[i].stageFlags = buffers_bits;
    bindings[i].pImmutableSamplers = nullptr; // Optional
  }

  for (uint32_t i = uniform_buffers_count; i < uniform_buffers_count + textures_count; ++i) {
    VkDescriptorSetLayoutBinding uboLayoutBinding = {};
    bindings[i].binding = i;
    bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[i].descriptorCount = 1;
    bindings[i].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    bindings[i].pImmutableSamplers = nullptr; // Optional
  }

  VkDescriptorSetLayoutCreateInfo layoutInfo = {};
  layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
  layoutInfo.pBindings = bindings.data();

  VkDescriptorSetLayout descriptorSetLayout = {};
  VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout));
  return descriptorSetLayout;
}

void RD_Vulkan::createDescriptorSetLayout() {
  gbufferDescriptorSetLayout = create_descriptors_set_layout(device, 1, 1, VK_SHADER_STAGE_VERTEX_BIT);
  shadowMapDescriptorSetLayout = create_descriptors_set_layout(device, 1, 0, VK_SHADER_STAGE_VERTEX_BIT);
  resolveDescriptorSetLayout = create_descriptors_set_layout(device, 2, 4, VK_SHADER_STAGE_FRAGMENT_BIT);
  postprocessDescriptorSetLayout = create_descriptors_set_layout(device, 0, 1, VK_SHADER_STAGE_FRAGMENT_BIT);
}

void RD_Vulkan::createDescriptorPool() {
  {
    std::array<VkDescriptorPoolSize, 2> poolSizes = {};
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSizes[0].descriptorCount = 1;
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSizes[1].descriptorCount = 1;

    VkDescriptorPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    poolInfo.maxSets = 1;

    VK_CHECK_RESULT(vkCreateDescriptorPool(device, &poolInfo, nullptr, &resolveDescriptorPool));
  }

  {
    std::array<VkDescriptorPoolSize, 1> poolSizes = {};
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSizes[0].descriptorCount = static_cast<uint32_t>(swapChainImages.size());

    VkDescriptorPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    poolInfo.maxSets = static_cast<uint32_t>(swapChainImages.size());

    VK_CHECK_RESULT(vkCreateDescriptorPool(device, &poolInfo, nullptr, &postprocessDescriptorPool));
  }
}

void RD_Vulkan::createDescriptorSets() {
  {
    VkDescriptorSetAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = resolveDescriptorPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &resolveDescriptorSetLayout;

    VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, &resolveDescriptorSets));

    std::vector<VkWriteDescriptorSet> descriptorWrites(2);

    std::array<VkDescriptorBufferInfo, 2> buffersInfo;
    buffersInfo[0].buffer = lightsBuffer;
    buffersInfo[0].offset = 0;
    buffersInfo[0].range = sizeof(DirectLight);

    buffersInfo[1].buffer = resolveConstants;
    buffersInfo[1].offset = 0;
    buffersInfo[1].range = sizeof(float4x4);

    descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[0].dstSet = resolveDescriptorSets;
    descriptorWrites[0].dstBinding = 0;
    descriptorWrites[0].dstArrayElement = 0;
    descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    descriptorWrites[0].descriptorCount = buffersInfo.size();
    descriptorWrites[0].pBufferInfo = buffersInfo.data();

    VkDescriptorImageInfo colorImageInfo = {};
    colorImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    colorImageInfo.imageView = colorImageView;
    colorImageInfo.sampler = colorImageSampler;

    VkDescriptorImageInfo normalImageInfo = {};
    normalImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    normalImageInfo.imageView = normalImageView;
    normalImageInfo.sampler = normalImageSampler;

    VkDescriptorImageInfo depthImageInfo = {};
    depthImageInfo.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
    depthImageInfo.imageView = depthImageView;
    depthImageInfo.sampler = depthImageSampler;

    VkDescriptorImageInfo shadowMapImageInfo = {};
    shadowMapImageInfo.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
    shadowMapImageInfo.imageView = shadowMapImageView;
    shadowMapImageInfo.sampler = shadowMapImageSampler;

    std::array<VkDescriptorImageInfo, 4> imagesInfo = { colorImageInfo, normalImageInfo, depthImageInfo, shadowMapImageInfo };

    descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[1].dstSet = resolveDescriptorSets;
    descriptorWrites[1].dstBinding = 2;
    descriptorWrites[1].dstArrayElement = 0;
    descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    descriptorWrites[1].descriptorCount = imagesInfo.size();
    descriptorWrites[1].pImageInfo = imagesInfo.data();

    vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
  }

  {
    std::vector<VkDescriptorSetLayout> layouts(swapChainImages.size(), postprocessDescriptorSetLayout);
    VkDescriptorSetAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = postprocessDescriptorPool;
    allocInfo.descriptorSetCount = static_cast<uint32_t>(swapChainImages.size());
    allocInfo.pSetLayouts = layouts.data();

    postprocessDescriptorSets.resize(swapChainImages.size());
    VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, postprocessDescriptorSets.data()));

    for (size_t i = 0; i < swapChainImages.size(); i++) {
      std::vector<VkWriteDescriptorSet> descriptorWrites(1);

      VkDescriptorImageInfo frameImageInfo = {};
      frameImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
      frameImageInfo.imageView = frameMipchainImageView;
      frameImageInfo.sampler = frameMipchainImageSampler;

      std::array<VkDescriptorImageInfo, 1> imagesInfo = { frameImageInfo };

      descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      descriptorWrites[0].dstSet = postprocessDescriptorSets[i];
      descriptorWrites[0].dstBinding = 0;
      descriptorWrites[0].dstArrayElement = 0;
      descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
      descriptorWrites[0].descriptorCount = imagesInfo.size();
      descriptorWrites[0].pImageInfo = imagesInfo.data();

      vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
    }
  }
}

VkDescriptorImageInfo RD_Vulkan::Texture::getDescriptor() const {
  VkDescriptorImageInfo imageInfo = {};
  imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  imageInfo.imageView = textureImageView;
  imageInfo.sampler = textureSampler;
  return imageInfo;
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
  samplerInfo.minLod = 0;
  samplerInfo.maxLod = static_cast<float>(mipLevels);

  VK_CHECK_RESULT(vkCreateSampler(deviceRef, &samplerInfo, nullptr, &textureSampler));
}

void RD_Vulkan::createColorSampler() {
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
  samplerInfo.maxLod = 1;

  VK_CHECK_RESULT(vkCreateSampler(device, &samplerInfo, nullptr, &colorImageSampler));

  VK_CHECK_RESULT(vkCreateSampler(device, &samplerInfo, nullptr, &normalImageSampler));

  VK_CHECK_RESULT(vkCreateSampler(device, &samplerInfo, nullptr, &depthImageSampler));

  VK_CHECK_RESULT(vkCreateSampler(device, &samplerInfo, nullptr, &shadowMapImageSampler));

  VK_CHECK_RESULT(vkCreateSampler(device, &samplerInfo, nullptr, &frameImageSampler));

  samplerInfo.maxLod = screenMipLevels;
  VK_CHECK_RESULT(vkCreateSampler(device, &samplerInfo, nullptr, &frameMipchainImageSampler));
}

void RD_Vulkan::createDefaultTexture() {
  const uint8_t CHANNELS_COUNT = 4;
  const uint8_t WHITE_COLOR[CHANNELS_COUNT] = {0xFF, 0xFF, 0xFF, 0xFF};
  defaultTexture = std::make_unique<Texture>(device, 1, 1, WHITE_COLOR);
}

void RD_Vulkan::createDepthResources() {
  VkFormat depthFormat = findDepthFormat();

  BufferManager::get().createImage(swapChainExtent.width, swapChainExtent.height, 1, depthFormat, VK_IMAGE_TILING_OPTIMAL,
    VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, depthImage, depthImageMemory);
  depthImageView = BufferManager::get().createImageView(depthImage, depthFormat, VK_IMAGE_ASPECT_DEPTH_BIT, 1);

  BufferManager::get().transitionImageLayout(depthImage, depthFormat, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL, 1);

  BufferManager::get().createImage(SHADOW_MAP_RESOLUTION, SHADOW_MAP_RESOLUTION, 1, depthFormat, VK_IMAGE_TILING_OPTIMAL,
    VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, shadowMapImage, shadowMapImageMemory);
  shadowMapImageView = BufferManager::get().createImageView(shadowMapImage, depthFormat, VK_IMAGE_ASPECT_DEPTH_BIT, 1);

  BufferManager::get().transitionImageLayout(shadowMapImage, depthFormat, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL, 1);
}

static uint32_t log2i(uint32_t val) {
  uint32_t res = 0;
  while (val > 1) {
    res++;
    val >>= 1;
  }
  return res;
}

void RD_Vulkan::createColorResources() {
  VkFormat colorFormat = swapChainImageFormat;
  screenMipLevels = log2i(min(swapChainExtent.width, swapChainExtent.height)) + 1;

  BufferManager::get().createImage(swapChainExtent.width, swapChainExtent.height, 1, colorFormat, VK_IMAGE_TILING_OPTIMAL,
    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, colorImage, colorImageMemory);
  colorImageView = BufferManager::get().createImageView(colorImage, colorFormat, VK_IMAGE_ASPECT_COLOR_BIT, 1);

  BufferManager::get().transitionImageLayout(colorImage, colorFormat, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1);

  VkFormat normalFormat = VK_FORMAT_R32G32B32A32_SFLOAT;
  BufferManager::get().createImage(swapChainExtent.width, swapChainExtent.height, 1, normalFormat, VK_IMAGE_TILING_OPTIMAL,
    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, normalImage, normalImageMemory);
  normalImageView = BufferManager::get().createImageView(normalImage, normalFormat, VK_IMAGE_ASPECT_COLOR_BIT, 1);

  BufferManager::get().transitionImageLayout(normalImage, normalFormat, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1);

  VkFormat frameFormat = VK_FORMAT_R32G32B32A32_SFLOAT;
  BufferManager::get().createImage(swapChainExtent.width, swapChainExtent.height, 1, frameFormat, VK_IMAGE_TILING_OPTIMAL,
    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, frameImage, frameImageMemory);
  frameImageView = BufferManager::get().createImageView(frameImage, frameFormat, VK_IMAGE_ASPECT_COLOR_BIT, 1);

  BufferManager::get().transitionImageLayout(frameImage, frameFormat, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1);

  BufferManager::get().createImage(swapChainExtent.width, swapChainExtent.height, screenMipLevels, frameFormat, VK_IMAGE_TILING_OPTIMAL,
    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, frameMipchainImage, frameMipchainImageMemory);
  frameMipchainImageView = BufferManager::get().createImageView(frameMipchainImage, frameFormat, VK_IMAGE_ASPECT_COLOR_BIT, screenMipLevels);

  BufferManager::get().transitionImageLayout(frameMipchainImage, frameFormat, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, screenMipLevels);

  createColorSampler();
}

VkFormat RD_Vulkan::BufferManager::findSupportedFormat(const std::vector<VkFormat>& candidates, VkImageTiling tiling, VkFormatFeatureFlags features) {
  for (VkFormat format : candidates) {
    VkFormatProperties props;
    vkGetPhysicalDeviceFormatProperties(physicalDeviceRef, format, &props);
    if (tiling == VK_IMAGE_TILING_LINEAR && (props.linearTilingFeatures & features) == features) {
      return format;
    }
    else if (tiling == VK_IMAGE_TILING_OPTIMAL && (props.optimalTilingFeatures & features) == features) {
      return format;
    }
  }

  throw std::runtime_error("failed to find supported format!");
}

VkFormat RD_Vulkan::findDepthFormat() {
  return BufferManager::get().findSupportedFormat(
    { VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT },
    VK_IMAGE_TILING_OPTIMAL,
    VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT
  );
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
  createShadowMapRenderPass();
  createGbufferRenderPass();
  createResolveRenderPass();
  createPostprocessRenderPass();
  createDescriptorSetLayout();
  createPipelines();
  createCommandPool();
  BufferManager::get().init(physicalDevice, device, commandPool, graphicsQueue);
  createColorResources();
  createDepthResources();
  createBuffers();
  createDescriptorPool();
  createDescriptorSets();
  createFramebuffers();
  createDefaultTexture();
  createSyncObjects();

  camFov       = 45.0f;
  camNearPlane = 0.1f;
  camFarPlane  = 1000.0f;
  camUp[0] = 0; camUp[1] = 1; camUp[2] = 0;

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

  vkDestroyDescriptorSetLayout(device, resolveDescriptorSetLayout, nullptr);
  vkDestroyDescriptorSetLayout(device, postprocessDescriptorSetLayout, nullptr);
  vkDestroyDescriptorSetLayout(device, gbufferDescriptorSetLayout, nullptr);
  vkDestroyDescriptorSetLayout(device, shadowMapDescriptorSetLayout, nullptr);

  vkDestroyBuffer(device, globalVertexBuffer, nullptr);
  vkFreeMemory(device, globalVertexBufferMemory, nullptr);
  vkDestroyBuffer(device, globalIndexBuffer, nullptr);
  vkFreeMemory(device, globalIndexBufferMemory, nullptr);

  vkDestroyBuffer(device, matricesBuffer, nullptr);
  vkFreeMemory(device, matricesBufferMemory, nullptr);

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

  if (a_texId >= textures.size()) {
    textures.resize(a_texId + 1);
  }

  if (bpp == sizeof(float4)) {
    textures[a_texId] = std::make_unique<Texture>(device, w, h, reinterpret_cast<const float*>(a_data));
  } else if (bpp == sizeof(uint32_t)) {
    textures[a_texId] = std::make_unique<Texture>(device, w, h, reinterpret_cast<const uint8_t*>(a_data));
  } else {
    throw "Unsupported texture format";
  }

  return true;
}

static float3 parse_color(const pugi::char_t* s) {
  float3 color;
  std::wstringstream input(s);
  input >> color.x >> color.y >> color.z;
  return color;
}


bool RD_Vulkan::UpdateMaterial(int32_t a_matId, pugi::xml_node a_materialNode)
{
  pugi::xml_node clrNode = a_materialNode.child(L"diffuse").child(L"color");
  pugi::xml_node texNode = clrNode.child(L"texture");
  pugi::xml_node mtxNode = a_materialNode.child(L"diffuse").child(L"sampler").child(L"matrix");

  bool isEmission = clrNode == nullptr;
  if (isEmission)
    clrNode = a_materialNode.child(L"emission").child(L"color"); // no diffuse color ? => draw emission color instead!

  if (clrNode != nullptr)
  {
    const wchar_t* clrStr = nullptr;
    
    if (clrNode.attribute(L"val") != nullptr)
      clrStr = clrNode.attribute(L"val").as_string();
    else
      clrStr = clrNode.text().as_string();

    if (!std::wstring(clrStr).empty())
    {
      float3 color = parse_color(clrStr);

      m_diffColors[a_matId * 3 + 0] = color.x;
      m_diffColors[a_matId * 3 + 1] = color.y;
      m_diffColors[a_matId * 3 + 2] = color.z;
    }
  }

  if (materials.size() <= a_matId) {
    materials.resize(a_matId + 1);
  }

  float4& color = materials[a_matId].color;
  color.x = m_diffColors[a_matId * 3 + 0];
  color.y = m_diffColors[a_matId * 3 + 1];
  color.z = m_diffColors[a_matId * 3 + 2];
  color.w = 0;
  if (isEmission) {
    float emissionMult = max(max(color.x, color.y), color.z);
    color /= emissionMult;
    color.w = emissionMult;
  }

  if (texNode != nullptr) {
    m_diffTexId[a_matId] = texNode.attribute(L"id").as_int();
    materials[a_matId].textureIdx = m_diffTexId[a_matId];
  } else {
    m_diffTexId[a_matId] = -1;
    materials[a_matId].textureIdx = -1;
  }

  return true;
}

bool RD_Vulkan::UpdateLight(int32_t a_lightIdId, pugi::xml_node a_lightNode)
{
  auto lightType = a_lightNode.attribute(L"type");
  if (lightType.as_string() == std::wstring(L"directional")) {
    DirectLightTemplate newTemplate;
    auto intencity = a_lightNode.child(L"intensity");
    newTemplate.color = parse_color(intencity.child(L"color").attribute(L"val").as_string());
    newTemplate.color *= intencity.child(L"multiplier").attribute(L"val").as_float();
    auto sizeNode = a_lightNode.child(L"size");
    newTemplate.innerRadius = sizeNode.attribute(L"inner_radius").as_float();
    newTemplate.outerRadius = sizeNode.attribute(L"outer_radius").as_float();
    directLightLib[a_lightIdId] = newTemplate;
  } else {
    std::wstring type = lightType.as_string();
    std::string castedType(type.begin(), type.end());
    std::cout << "Light " << a_lightIdId << " not processed. Light type: " << castedType << std::endl;
  }
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

  if (!std::wstring(camPosStr).empty())
  {
    std::wstringstream input(camPosStr);
    input >> camPos[0] >> camPos[1] >> camPos[2];
  }

  if (!std::wstring(camLAtStr).empty())
  {
    std::wstringstream input(camLAtStr);
    input >> camLookAt[0] >> camLookAt[1] >> camLookAt[2];
  }

  if (!std::wstring(camUpStr).empty())
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
  if (inited) {
    return true;
  }
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

  std::vector<uint32_t> meshIndices(a_input.indices, a_input.indices + a_input.triNum * 3);
  std::vector<Vertex> meshVertices;
  for (int i = 0; i < a_input.vertNum; ++i) {
    float3 pos = { a_input.pos4f[4 * i], a_input.pos4f[4 * i + 1], a_input.pos4f[4 * i + 2] };
    float3 normal = { a_input.norm4f[4 * i], a_input.norm4f[4 * i + 1], a_input.norm4f[4 * i + 2] };
    float2 tc = { a_input.texcoord2f[2 * i], a_input.texcoord2f[2 * i + 1] };
    Vertex vertex = { pos, normal, tc };
    meshVertices.push_back(vertex);
  }

  vertices.insert(vertices.end(), meshVertices.begin(), meshVertices.end());
  const uint32_t indicesOffset = indices.size();
  indices.resize(indicesOffset + meshIndices.size());
  std::transform(meshIndices.begin(), meshIndices.end(), indices.begin() + indicesOffset, [indicesOffset](const uint32_t idx) { return idx + indicesOffset; });

  for (int matId = 0; matId < begins.size() - 1; ++matId) {
    const int matNum = a_input.triMatIndices[begins[matId]];
    StaticMesh subMesh;
    subMesh.indicesOffset = begins[matId] * 3 + indicesOffset;
    subMesh.incidesCount = (begins[matId + 1] - begins[matId]) * 3;
    subMesh.materialId = matNum;
    modelsLib[a_meshId].meshes.emplace_back(subMesh);
  }
  return true;
}


void RD_Vulkan::BeginScene(pugi::xml_node a_sceneNode)
{
  allRemapLists.clear();
  tableOffsetsAndSize.clear();

  pugi::xml_node remapLists = a_sceneNode.child(L"remap_lists");
  if (remapLists != nullptr)
  {
    for (pugi::xml_node listNode : remapLists.children())
    {
      const wchar_t* inputStr = listNode.attribute(L"val").as_string();
      const int listSize = listNode.attribute(L"size").as_int();
      std::wstringstream inStrStream(inputStr);

      tableOffsetsAndSize.emplace_back(int(allRemapLists.size()), listSize);

      for (int i = 0; i < listSize; i++)
      {
        if (inStrStream.eof())
          break;

        int data;
        inStrStream >> data;
        allRemapLists.push_back(data);
      }
    }
  }
  if ((screenShot || screenshotAndExit) && screenshotState == ScreenshotState::OFF) {
    screenshotState = ScreenshotState::REQUIRED;
    screenShot = false;
  }
}

void RD_Vulkan::createBuffers() {
  BufferManager::get().createBuffer(sizeof(directLights[0]), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, lightsBuffer, lightsBufferMemory);
  BufferManager::get().createBuffer(sizeof(float4x4), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, resolveConstants, resolveConstantsMemory);
}

void RD_Vulkan::EndScene()
{
  if (reloadShaders) {
    recreateShaders();
    reloadShaders = false;
  }
  if (inited) {
    return;
  }
  inited = true;
  void* data;
  if (hasDirectLight)
  {
    vkMapMemory(device, lightsBufferMemory, 0, directLights.size() * sizeof(directLights[0]), 0, &data);
    memcpy(data, directLights.data(), directLights.size() * sizeof(directLights[0]));
    vkUnmapMemory(device, lightsBufferMemory);
  }

  createVertexBuffer();
  createIndexBuffer();
  BufferManager::get().createBuffer(sizeof(matrices[0]) * matrices.size(), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, matricesBuffer, matricesBufferMemory);
  vkMapMemory(device, matricesBufferMemory, 0, sizeof(matrices[0]) * matrices.size(), 0, &data);
  memcpy(data, matrices.data(), sizeof(matrices[0]) * matrices.size());
  vkUnmapMemory(device, matricesBufferMemory);

  {
    std::array<VkDescriptorPoolSize, 2> poolSizes = {};
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSizes[0].descriptorCount = static_cast<uint32_t>(materials.size());
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSizes[1].descriptorCount = static_cast<uint32_t>(materials.size());

    VkDescriptorPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    poolInfo.maxSets = static_cast<uint32_t>(materials.size());

    VK_CHECK_RESULT(vkCreateDescriptorPool(device, &poolInfo, nullptr, &gbufferDescriptorPool));
  }

  {
    VkDescriptorPoolSize poolSizes = {};
    poolSizes.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSizes.descriptorCount = 1;

    VkDescriptorPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSizes;
    poolInfo.maxSets = 1;

    VK_CHECK_RESULT(vkCreateDescriptorPool(device, &poolInfo, nullptr, &shadowMapDescriptorPool));
  }

  materialsLib.resize(materials.size());
  std::vector<VkDescriptorSetLayout> layouts(materialsLib.size(), gbufferDescriptorSetLayout);
  VkDescriptorSetAllocateInfo allocInfo = {};
  allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  allocInfo.descriptorPool = gbufferDescriptorPool;
  allocInfo.descriptorSetCount = layouts.size();
  allocInfo.pSetLayouts = layouts.data();
  VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, materialsLib.data()));
  for (uint32_t i = 0; i < materials.size(); ++i) {
    std::vector<VkWriteDescriptorSet> descriptorWrites(2);

    std::array<VkDescriptorBufferInfo, 1> buffersInfo;
    buffersInfo[0].buffer = matricesBuffer;
    buffersInfo[0].offset = 0;
    buffersInfo[0].range = sizeof(matrices[0]) * matrices.size();

    descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[0].dstSet = materialsLib[i];
    descriptorWrites[0].dstBinding = 0;
    descriptorWrites[0].dstArrayElement = 0;
    descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    descriptorWrites[0].descriptorCount = buffersInfo.size();
    descriptorWrites[0].pBufferInfo = buffersInfo.data();

    Texture* tex = (materials[i].textureIdx == -1 ? defaultTexture : textures[materials[i].textureIdx]).get();
    VkDescriptorImageInfo imageInfo = tex->getDescriptor();

    descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[1].dstSet = materialsLib[i];
    descriptorWrites[1].dstBinding = 1;
    descriptorWrites[1].dstArrayElement = 0;
    descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    descriptorWrites[1].descriptorCount = 1;
    descriptorWrites[1].pImageInfo = &imageInfo;

    vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
  }

  {
    VkDescriptorSetAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = shadowMapDescriptorPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &shadowMapDescriptorSetLayout;
    VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, &materialsShadowDs));
    VkWriteDescriptorSet descriptorWrites = {};

    std::array<VkDescriptorBufferInfo, 1> buffersInfo;
    buffersInfo[0].buffer = matricesBuffer;
    buffersInfo[0].offset = 0;
    buffersInfo[0].range = sizeof(matrices[0]) * matrices.size();

    descriptorWrites.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites.dstSet = materialsShadowDs;
    descriptorWrites.dstBinding = 0;
    descriptorWrites.dstArrayElement = 0;
    descriptorWrites.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    descriptorWrites.descriptorCount = buffersInfo.size();
    descriptorWrites.pBufferInfo = buffersInfo.data();

    vkUpdateDescriptorSets(device, 1, &descriptorWrites, 0, nullptr);
  }

  createCommandBuffers();
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

  if (screenshotState == ScreenshotState::IN_PROGRESS && imageIndex == screenshotFrameIdx) {
    VkDeviceSize imageSize = m_width * m_height * 4;
    std::vector<uint32_t> imageData(imageSize / 4);

    void* data;
    vkMapMemory(device, screenshotBufferMemory, 0, imageSize, 0, &data);
    memcpy(imageData.data(), data, static_cast<size_t>(imageSize));
    vkUnmapMemory(device, screenshotBufferMemory);

    vkDestroyBuffer(device, screenshotBuffer, nullptr);
    vkFreeMemory(device, screenshotBufferMemory, nullptr);

    for (uint32_t i = 0; i < imageData.size(); ++i) {
      imageData[i] = ((imageData[i] & 0xFF) << 16) | (((imageData[i] >> 8) & 0xFF) << 8) | (((imageData[i] >> 16) & 0xFF)) | (((imageData[i] >> 24) & 0xFF) << 24);
    }
    const uint32_t rowSize = m_width * 4;
    std::vector<uint32_t> rowBuffer(rowSize);
    for (uint32_t i = 0; i < m_height / 2; ++i) {
      memcpy(rowBuffer.data(), &imageData[i * m_width], rowSize);
      memcpy(&imageData[i * m_width], &imageData[(m_height - i - 1) * m_width], rowSize);
      memcpy(&imageData[(m_height - i - 1) * m_width], rowBuffer.data(), rowSize);
    }
    auto end = std::chrono::system_clock::now();
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);
    std::tm* time = std::localtime(&end_time);

    char dateBuffer[1024];
    sprintf(dateBuffer, "%d.%02d.%02d_%02d.%02d.%02d.bmp", time->tm_year + 1900, time->tm_mon + 1, time->tm_mday, time->tm_hour, time->tm_min, time->tm_sec);

    SaveBMP(dateBuffer, imageData.data(), m_width, m_height);
    std::cout << "Screenshot saved to file " << dateBuffer << std::endl;

    screenshotState = ScreenshotState::OFF;
    if (screenshotAndExit) {
      exit(0);
    }
  }

  updateUniformBuffer(imageIndex);
  prepareCommandBuffers(imageIndex);

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
  }
  else if (result != VK_SUCCESS) {
    throw std::runtime_error("failed to present swap chain image!");
  }

  currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
}

float4x4 orthographic_projection_matrix(float left_plane, float right_plane, float bottom_plane, float top_plane, float near_plane, float far_plane) {
  float4x4 orthographic_projection_matrix;
  orthographic_projection_matrix.row[0] = float4(2.0f / (right_plane - left_plane), 0.0f, 0.0f, 0.0f);
  orthographic_projection_matrix.row[1] = float4(0.0f, 2.0f / (bottom_plane - top_plane), 0.0f, 0.0f);
  orthographic_projection_matrix.row[2] = float4(0.0f, 0.0f, 1.0f / (near_plane - far_plane), 0.0f);
  orthographic_projection_matrix.row[3] = float4(-(right_plane + left_plane) / (right_plane - left_plane), -(bottom_plane + top_plane) / (bottom_plane - top_plane), near_plane / (near_plane - far_plane), 1.0f);
  return orthographic_projection_matrix;
}

void RD_Vulkan::updateUniformBuffer(uint32_t current_image) {
  const float3 eye(camPos[0], camPos[1], camPos[2]);
  const float3 center(camLookAt[0], camLookAt[1], camLookAt[2]);
  const float3 up(camUp[0], camUp[1], camUp[2]);
  const float4x4 view = lookAtTransposed(eye, center, up);

  const float aspect = float(m_width) / float(m_height);
  float4x4 proj = projectionMatrixTransposed(camFov, aspect, camNearPlane, camFarPlane);
  proj.M(1, 1) *= -1.f;

  globtm = mul(view, proj);
  const float4x4 invGlobtm = inverse4x4(globtm);
  std::array<float4, 4> viewVecsData = { invGlobtm.row[0], invGlobtm.row[1], invGlobtm.row[2], invGlobtm.row[3] };
  void* data;
  vkMapMemory(device, resolveConstantsMemory, 0, sizeof(viewVecsData), 0, &data);
  memcpy(data, viewVecsData.data(), sizeof(viewVecsData));
  vkUnmapMemory(device, resolveConstantsMemory);

  if (hasDirectLight)
  {
    float4x4 pr = orthographic_projection_matrix(-directLights[0].outerRadius, directLights[0].outerRadius, -directLights[0].outerRadius, directLights[0].outerRadius, 0.1, 200.0);
    float4x4 view = lookAtTransposed(directLights[0].position, directLights[0].direction + directLights[0].position, float3(0, 1, 0));

    lighttm = mul(view, pr);
  }
}

static inline void mat4x4_transpose(float M[16], const float N[16])
{
  const uint32_t SIDE = 4;
  for (int j = 0; j < SIDE; j++)
  {
    for (int i = 0; i < SIDE; i++)
      M[i * SIDE + j] = N[j * SIDE + i];
  }
}

void RD_Vulkan::InstanceMeshes(int32_t a_mesh_id, const float* a_matrices, int32_t a_instNum, const int* a_lightInstId, const int* a_remapId, const int* a_realInstId)
{
  if (inited) {
    return;
  }
  std::vector<float4x4> models(a_instNum, a_matrices);
  for (int32_t i = 0; i < a_instNum; i++)
  {
    float matrixT2[16];
    mat4x4_transpose(matrixT2, (float*)(a_matrices + i * 16));
    models[i] = float4x4(matrixT2);
  }

  StaticModelInstances modelInstance;
  for (auto mesh : modelsLib[a_mesh_id].meshes) {
    StaticMeshInstances meshInst;
    meshInst.matricesOffset = matrices.size();
    meshInst.matricesCount = a_instNum;
    meshInst.mesh = mesh;
    if (a_remapId[0] != -1) {
      const int jBegin = tableOffsetsAndSize[a_remapId[0]].x;
      const int jEnd = tableOffsetsAndSize[a_remapId[0]].x + tableOffsetsAndSize[a_remapId[0]].y;
      for (int j = jBegin; j < jEnd; j += 2) {
        if (allRemapLists[j] == meshInst.mesh.materialId) {
          meshInst.mesh.materialId = allRemapLists[j + 1];
          break;
        }
      }
    }
    modelInstance.parts.push_back(meshInst);
  }
  matrices.insert(matrices.end(), models.begin(), models.end());
  modelInstances.push_back(modelInstance);
}


void RD_Vulkan::InstanceLights(int32_t a_light_id, const float* a_matrix, pugi::xml_node* a_custAttrArray, int32_t a_instNum, int32_t a_lightGroupId)
{
  if (inited) {
    return;
  }

  int lightId = a_custAttrArray->attribute(L"light_id").as_int();
  if (directLightLib.count(lightId)) {
    for (uint32_t i = 0; i < a_instNum; ++i) {
      DirectLight lightToAdd;
      lightToAdd.color = directLightLib[lightId].color;
      lightToAdd.innerRadius = directLightLib[lightId].innerRadius;
      lightToAdd.outerRadius = directLightLib[lightId].outerRadius;
      float4x4 matrix(a_matrix + i * 16);
      matrix = transpose(matrix);
      lightToAdd.direction = -to_float3(matrix.row[1]);
      lightToAdd.position = to_float3(matrix.row[3]);
      directLights.push_back(lightToAdd);
      hasDirectLight = true;
    }
  }
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

void RD_Vulkan::recreateShaders() {
  destroyPipelines();
  createPipelines();
}

void RD_Vulkan::destroyPipelines() {
  vkDestroyPipeline(device, shadowMapPipeline, nullptr);
  vkDestroyPipeline(device, graphicsPipeline, nullptr);
  vkDestroyPipeline(device, resolvePipeline, nullptr);
  vkDestroyPipeline(device, postprocessPipeline, nullptr);
  vkDestroyPipelineLayout(device, shadowMapPipelineLayout, nullptr);
  vkDestroyPipelineLayout(device, gbufferPipelineLayout, nullptr);
  vkDestroyPipelineLayout(device, resolvePipelineLayout, nullptr);
  vkDestroyPipelineLayout(device, postprocessPipelineLayout, nullptr);
}
