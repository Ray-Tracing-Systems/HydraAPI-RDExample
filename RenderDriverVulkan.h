#pragma once

#include "HydraRenderDriverAPI.h"

#ifdef WIN32
  #include <windows.h>
#endif

#include <vulkan/vulkan.h>

#include <cassert>

#include <array>
#include <map>
#include <memory>
#include <optional>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <unordered_map>

#include "LiteMath.h"

using HydraLiteMath::float3;

#include "lights.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define VK_CHECK_RESULT(f) 													\
{																										\
    VkResult res = (f);															\
    if (res != VK_SUCCESS)													\
    {																								\
        printf("Fatal : VkResult is %d in %s at line %d\n", res,  __FILE__, __LINE__); \
        assert(res == VK_SUCCESS);									\
    }																								\
}


struct Vertex;
class UniformBufferObject {
  HydraLiteMath::float4x4 model;
  HydraLiteMath::float4x4 view;
  HydraLiteMath::float4x4 proj;
  HydraLiteMath::float4x4 result;
  void updateResult() {
    result = mul(model, mul(view, proj));
  }

public:
  void setModel(const HydraLiteMath::float4x4& m) {
    model = m;
  }

  void setView(const HydraLiteMath::float3& eye, const HydraLiteMath::float3& center, const HydraLiteMath::float3& up) {
    view = lookAtTransposed(eye, center, up);
  }

  void setView(const HydraLiteMath::float4x4& m) {
    view = m;
  }

  void setProj(float camFov, float aspect, float camNearPlane, float camFarPlane) {
    proj = HydraLiteMath::projectionMatrixTransposed(camFov, aspect, camNearPlane, camFarPlane);
    proj.M(1, 1) *= -1.f;
  }

  void setProj(const HydraLiteMath::float4x4& m) {
    proj = m;
  }

  const HydraLiteMath::float4x4& getResultRef() {
    updateResult();
    return result;
  }

  const HydraLiteMath::float4x4& getModel() const {
    return model;
  }
};

enum class ScreenshotState {
  OFF,
  REQUIRED,
  IN_PROGRESS
};

struct RD_Vulkan : public IHRRenderDriver
{
  RD_Vulkan();
  ~RD_Vulkan();

  void GetRenderDriverName(std::wstring &name) override { name = std::wstring(L"vulkan");};
  void              ClearAll() override;
  HRDriverAllocInfo AllocAll(HRDriverAllocInfo a_info) override;
       
  bool UpdateImage(int32_t a_texId, int32_t w, int32_t h, int32_t bpp, const void* a_data, pugi::xml_node a_texNode) override;
  bool UpdateMaterial(int32_t a_matId, pugi::xml_node a_materialNode) override;
  bool UpdateLight(int32_t a_lightIdId, pugi::xml_node a_lightNode) override;
  bool UpdateMesh(int32_t a_meshId, pugi::xml_node a_meshNode, const HRMeshDriverInput& a_input, const HRBatchInfo* a_batchList, int32_t listSize) override;
       
  bool UpdateImageFromFile(int32_t, const wchar_t*, pugi::xml_node) override { return false; }
  bool UpdateMeshFromFile(int32_t, pugi::xml_node, const wchar_t*) override { return false; }

       
  bool UpdateCamera(pugi::xml_node a_camNode) override;
  bool UpdateSettings(pugi::xml_node a_settingsNode) override;

  /////////////////////////////////////////////////////////////////////////////////////////////

  void BeginScene(pugi::xml_node a_sceneNode) override;
  void EndScene() override;
  void InstanceMeshes(int32_t a_mesh_id, const float* a_matrices, int32_t a_instNum, const int* a_lightInstId, const int* a_remapId, const int* a_realInstId) override;
  void InstanceLights(int32_t a_light_id, const float* a_matrix, pugi::xml_node* a_custAttrArray, int32_t a_instNum, int32_t a_lightGroupId) override;

  void Draw() override;

  HRRenderUpdateInfo HaveUpdateNow(int a_maxRaysPerPixel) override;

  void GetFrameBufferHDR(int32_t w, int32_t h, float*   a_out, const wchar_t* a_layerName) override;
  void GetFrameBufferLDR(int32_t w, int32_t h, int32_t* a_out) override;

  void GetGBufferLine(int32_t, HRGBufferPixel*, int32_t, int32_t, const std::unordered_set<int32_t>&) override {}

  const HRRenderDeviceInfoListElem* DeviceList() const override { return nullptr; } //#TODO: implement quering GPU info bu glGetString(GL_VENDOR) and e.t.c.
  bool EnableDevice(int32_t, bool) override { return true; }

protected:
  struct QueueFamilyIndices {
    uint32_t graphicsFamily = 0;
    uint32_t presentFamily = 0;
  };

  class Texture {
    template<typename T>
    struct TypeToFormat
    {
      enum {
        format = std::is_same<float, T>::value ? VK_FORMAT_R32G32B32A32_SFLOAT : VK_FORMAT_R8G8B8A8_UNORM
      };
    };

    VkDevice deviceRef;

    VkImage textureImage;
    VkDeviceMemory textureImageMemory;
    VkImageView textureImageView;
    uint32_t mipLevels;

    template<typename T>
    void createTextureImage(uint32_t width, uint32_t height, const T* image, VkFormat format);
    void createTextureImageView(VkFormat format);
  public:
    template<typename T>
    Texture(VkDevice dev, uint32_t width, uint32_t height, const T* image) : deviceRef(dev) {
      createTextureImage(width, height, image, VkFormat(TypeToFormat<T>::format));
      createTextureImageView(VkFormat(TypeToFormat<T>::format));
    }

    ~Texture();

    VkDescriptorImageInfo getDescriptor() const;
  };

  struct Material {
    int textureIdx = -1;
    HydraLiteMath::float4 color;
    HydraLiteMath::float4 emission;
  };

  class BufferManager {
    VkPhysicalDevice physicalDeviceRef;
    VkDevice deviceRef;
    VkCommandPool commandPoolRef;
    VkQueue queueRef;

  public:
    void init(VkPhysicalDevice physicalDevice, VkDevice device, VkCommandPool commandPool, VkQueue queue) {
      physicalDeviceRef = physicalDevice;
      deviceRef = device;
      commandPoolRef = commandPool;
      queueRef = queue;
    }
    static BufferManager& get() {
      static BufferManager instance;
      return instance;
    }

    void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory);
    void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size);
    void transitionImageLayout(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout, uint32_t mipLevels, uint32_t layers_count);
    void copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height);
    void copyImageToBuffer(VkImage image, VkBuffer buffer, uint32_t width, uint32_t height);
    void generateMipmaps(VkImage image, VkFormat imageFormat, int32_t texWidth, int32_t texHeight, uint32_t mipLevels);

    VkImageView createImageView(VkImage image, VkFormat format, VkImageAspectFlags aspectFlags, uint32_t mipLevels);
    VkImageView createTextureArrayView(VkImage image, VkFormat format, VkImageAspectFlags aspectFlags, uint32_t mipLevels, uint32_t base_layer, uint32_t layers_count);
    void createImage(uint32_t width, uint32_t height, uint32_t mip_levels, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage,
      VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& imageMemory);
    void createTextureArray(uint32_t width, uint32_t height, uint32_t mip_levels, uint32_t layers_count,
      VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage,
      VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& imageMemory);

    VkFormat findSupportedFormat(const std::vector<VkFormat>& candidates, VkImageTiling tiling, VkFormatFeatureFlags features);

    VkCommandBuffer beginSingleTimeCommands();
    void endSingleTimeCommands(VkCommandBuffer commandBuffer);
  private:
    BufferManager() {}
    ~BufferManager() {}

    class SingleTimeCommandsContext {
      VkCommandBuffer commandBuffer;
    public:
      SingleTimeCommandsContext() {
        commandBuffer = BufferManager::get().beginSingleTimeCommands();
      }
      ~SingleTimeCommandsContext() {
        BufferManager::get().endSingleTimeCommands(commandBuffer);
      }
      VkCommandBuffer getCB() const {
        return commandBuffer;
      }
    };
  };

  struct StaticMesh {
    uint32_t indicesOffset;
    uint32_t incidesCount;
    uint32_t materialId;
  };

  struct StaticMeshInstances {
    uint32_t matricesOffset;
    uint32_t matricesCount;
    StaticMesh mesh;
  };

  struct StaticModel {
    std::vector<StaticMesh> meshes;
  };

  struct StaticModelInstances {
    std::vector<StaticMeshInstances> parts;
  };

  struct PipelineConfig {
    std::string vertexShaderPath;
    std::optional<std::string> pixelShaderPath;
    bool hasVertexBuffer = false;
    VkRenderPass renderPass = {};
    VkDescriptorSetLayout descriptorSetLayout = {};
    uint32_t rtCount = 1;
    std::vector<VkPushConstantRange> pushConstants;
    VkPrimitiveTopology primitiveTopology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkCullModeFlagBits cullingBits = VK_CULL_MODE_NONE;

    uint32_t width, height;
  };

  struct ComputePipelineConfig {
    std::string shaderPath;
    std::vector<VkPushConstantRange> pushConstants;
    VkDescriptorSetLayout descriptorSetLayout = {};
  };

  struct DirectLightTemplate {
    HydraLiteMath::float3 color;
    float innerRadius; float outerRadius;
  };

  struct SpotLightTemplate {
    HydraLiteMath::float3 color;
    float innerCos; float outerCos;
  };

  void createInstance();
  void pickPhysicalDevice();
  void createLogicalDevice();
  void createSurface();
  void createSwapChain();
  void createImageViews();
  void createPipelines();
  VkPipeline createGraphicsPipeline(const PipelineConfig&, VkPipelineLayout& layout);
  VkPipeline createComputePipeline(const ComputePipelineConfig& config, VkPipelineLayout& layout);
  void createGbufferRenderPass();
  void createShadowMapRenderPass();
  void createResolveRenderPass();
  void createPostprocessRenderPass();
  void createFramebuffers();
  void createCommandPool();
  void createCommandBuffers();
  void createSyncObjects();
  void recreateSwapChain();
  void cleanupSwapChain();
  void createDefaultTexture();
  void createDepthResources();
  void createColorResources();
  void createBuffers();
  void createLightingBuffer();
  VkFormat findDepthFormat();
  QueueFamilyIndices GetQueueFamilyIndex();

  void createDescriptorSetLayout();
  void createDescriptorSets();
  void createDescriptorPool();
  void updateUniformBuffer(uint32_t current_image);
  void createVertexBuffer();
  void createIndexBuffer();
  void prepareDebugPoints();
  void prepareCommandBuffers(uint32_t current_image);
  void recreateShaders();
  void destroyPipelines();

  std::wstring m_libPath;

  // camera parameters
  //
  float camPos[3];
  float camLookAt[3];
  float camUp[3];

  float camFov;
  float camNearPlane;
  float camFarPlane;
  int   m_width;
  int   m_height;

  HydraLiteMath::float4x4 camWorldViewMartrixTransposed;
  HydraLiteMath::float4x4 camProjMatrixTransposed;
  bool                    camUseMatrices;

  std::vector<int>  allRemapLists;
  std::vector<HydraLiteMath::int2> tableOffsetsAndSize;

  std::vector<float> m_diffColors;
  std::vector<int>   m_diffTexId;
  VkInstance vk_inst;
  VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
  VkDevice device;
  VkSurfaceKHR surface;
  VkSwapchainKHR swapchain;
  static const int MAX_FRAMES_IN_FLIGHT = 2;
  std::array<VkSemaphore, MAX_FRAMES_IN_FLIGHT> imageAvailableSemaphores;
  std::array<VkSemaphore, MAX_FRAMES_IN_FLIGHT> renderFinishedSemaphores;
  std::array<VkFence, MAX_FRAMES_IN_FLIGHT> inFlightFences;
  VkQueue graphicsQueue;
  std::vector<VkCommandBuffer> commandBuffers;
  std::vector<VkImage> swapChainImages;
  std::vector<VkImageView> swapChainImageViews;
  std::vector<VkFramebuffer> swapChainFramebuffers;
  VkFramebuffer gbufferFramebuffer;
  std::vector<VkFramebuffer> shadowMapFramebuffers;
  VkFramebuffer resolveFramebuffer;
  VkFormat swapChainImageFormat;
  VkExtent2D swapChainExtent;
  VkDescriptorSetLayout shadowMapDescriptorSetLayout;
  VkDescriptorSetLayout gbufferDescriptorSetLayout;
  VkDescriptorSetLayout resolveDescriptorSetLayout;
  VkDescriptorSetLayout postprocessDescriptorSetLayout;
  VkDescriptorSetLayout debugPointsDescriptorSetLayout;
  VkDescriptorSetLayout initialLightingDescriptorSetLayout;
  VkDescriptorSetLayout lightBounceDescriptorSetLayout;
  VkDescriptorSetLayout nextBounceDescriptorSetLayout;
  VkPipelineLayout shadowMapPipelineLayout;
  VkPipelineLayout gbufferPipelineLayout;
  VkPipelineLayout debugPointsPipelineLayout;
  VkPipelineLayout resolvePipelineLayout;
  VkPipelineLayout postprocessPipelineLayout;
  VkPipelineLayout initialLightingPipelineLayout;
  VkPipelineLayout lightBouncePipelineLayout;
  VkPipelineLayout nextBouncePipelineLayout;
  VkRenderPass gbufferRenderPass;
  VkRenderPass shadowMapRenderPass;
  VkRenderPass resolveRenderPass;
  VkRenderPass postprocessRenderPass;
  VkPipeline shadowMapPipeline;
  VkPipeline graphicsPipeline;
  VkPipeline resolvePipeline;
  VkPipeline debugPointsPipeline;
  VkPipeline postprocessPipeline;
  VkPipeline initialLightingPipeline;
  VkPipeline lightBouncePipeline;
  VkPipeline nextBouncePipeline;
  VkCommandPool commandPool;

  VkImage depthImage;
  VkDeviceMemory depthImageMemory;
  VkImageView depthImageView;

  VkImage colorImage;
  VkDeviceMemory colorImageMemory;
  VkImageView colorImageView;

  VkImage normalImage;
  VkDeviceMemory normalImageMemory;
  VkImageView normalImageView;

  VkImage frameImage;
  VkDeviceMemory frameImageMemory;
  VkImageView frameImageView;

  VkImage frameMipchainImage;
  VkDeviceMemory frameMipchainImageMemory;
  VkImageView frameMipchainImageView;

  VkImage shadowMapImage;
  VkDeviceMemory shadowMapImageMemory;
  VkImageView shadowMapImageArrayView;
  std::vector<VkImageView> shadowMapImageLayerViews;
  const uint32_t SHADOW_MAP_RESOLUTION = 4096;

  VkDescriptorPool resolveDescriptorPool = {};
  VkDescriptorPool postprocessDescriptorPool = {};
  VkDescriptorPool gbufferDescriptorPool = {};
  VkDescriptorPool shadowMapDescriptorPool = {};
  VkDescriptorPool initialLightingDescriptorPool = {};
  VkDescriptorPool lightBounceDescriptorPool = {};
  VkDescriptorPool nextBounceDescriptorPool = {};
  VkDescriptorSet resolveDescriptorSets;
  std::vector<VkDescriptorSet> postprocessDescriptorSets;

  size_t currentFrame = 0;
  bool inited = false;
  std::vector<std::unique_ptr<Texture>> textures;
  std::unique_ptr<Texture> defaultTexture;
  std::vector<Material> materials;
  std::unordered_map<uint32_t, DirectLightTemplate> directLightLib;
  std::unordered_map<uint32_t, SpotLightTemplate> spotLightLib;
  std::vector<DirectLight> directLights;
  std::vector<SpotLight> spotLights;
  std::vector<HydraLiteMath::float4x4> lightsTM;
  std::unordered_map<int, StaticModel> modelsLib;
  std::vector<StaticModelInstances> modelInstances;
  std::vector<VkDescriptorSet> materialsLib;
  VkDescriptorSet materialsShadowDs;
  VkDescriptorSet initialLightingDs;
  VkDescriptorSet lightBounceDs;
  VkDescriptorSet nextBounceDs;
  VkBuffer resolveConstants;
  VkDeviceMemory resolveConstantsMemory;

  VkBuffer lightsBuffer;
  VkDeviceMemory lightsBufferMemory;

  VkBuffer globalVertexBuffer = {};
  VkDeviceMemory globalVertexBufferMemory = {};
  VkBuffer globalIndexBuffer = {};
  VkDeviceMemory globalIndexBufferMemory = {};

  VkBuffer debugPointsVertexBuffer = {};
  VkDeviceMemory debugPointsVertexBufferMemory = {};
  uint32_t debugPointsCount = 0;

  std::vector<Vertex> vertices;
  std::vector<uint32_t> indices;
  std::vector<HydraLiteMath::float4x4> matrices;

  VkBuffer matricesBuffer = {};
  VkDeviceMemory matricesBufferMemory = {};
  HydraLiteMath::float4x4 globtm;

  bool lightingInited = false;
  VkBuffer lightingBuffer;
  VkDeviceMemory lightingMemory;
  VkBuffer lightingWeightsBuffer;
  VkDeviceMemory lightingWeightsMemory;
  uint32_t lightingBufferSize;
  uint32_t lightingWeightsBufferSize;
  HydraLiteMath::uint4 gridSize;
  HydraLiteMath::float4 bmin;
  HydraLiteMath::float4 bmax;
  HydraLiteMath::float4 averageLighting;
  VkBuffer ffBuffer;
  VkDeviceMemory ffMemory;
  uint32_t ffBufferSize;
  VkBuffer ffOffsetsBuffer;
  VkDeviceMemory ffOffsetsMemory;
  uint32_t ffOffsetsBufferSize;
  uint32_t maxFFRow;
  VkBuffer currentBounceBuffer;
  VkDeviceMemory currentBounceMemory;
  uint32_t currentBounceBufferSize;
  VkBuffer finalLightingBuffer;
  VkDeviceMemory finalLightingMemory;
  uint32_t finalLightingBufferSize;
  VkBuffer colorsBuffer;
  VkDeviceMemory colorsMemory;
  VkBuffer idxToVoxelIdBuffer;
  VkDeviceMemory idxToVoxelIdMemory;
  uint32_t idxToVoxelIdBufferSize;
  VkBuffer voxelIdToIdxBuffer;
  VkDeviceMemory voxelIdToIdxMemory;
  uint32_t voxelIdToIdxBufferSize;

  uint32_t screenMipLevels = 0;
  ScreenshotState screenshotState = ScreenshotState::OFF;
  VkBuffer screenshotBuffer = {};
  VkDeviceMemory screenshotBufferMemory = {};
  uint32_t screenshotFrameIdx;

  using SamplerDescription = float; // Currently it is amount of mip levels, to be changed in future.
  class Core {
    std::unordered_map<SamplerDescription, VkSampler> samplers;
    VkDevice device;
  public:
    void init(VkDevice device_) {
      device = device_;
    }

    void close();

    VkSampler requestSampler(const SamplerDescription& desc);

    static Core& get() {
      static Core core;
      return core;
    }
  };
};
