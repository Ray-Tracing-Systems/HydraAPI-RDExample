#pragma once

#include "HydraRenderDriverAPI.h"

#ifdef WIN32
  #include <windows.h>
#endif

#include <vulkan/vulkan.h>

#include <array>
#include <map>
#include <memory>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <unordered_map>

#include "LiteMath.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

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
       
  bool UpdateImageFromFile(int32_t a_texId, const wchar_t* a_fileName, pugi::xml_node a_texNode) override { return false; }
  bool UpdateMeshFromFile(int32_t a_meshId, pugi::xml_node a_meshNode, const wchar_t* a_fileName) override { return false; }

       
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

  void GetGBufferLine(int32_t a_lineNumber, HRGBufferPixel* a_lineData, int32_t a_startX, int32_t a_endX, const std::unordered_set<int32_t>& a_shadowCatchers) override {}

  const HRRenderDeviceInfoListElem* DeviceList() const override { return nullptr; } //#TODO: implement quering GPU info bu glGetString(GL_VENDOR) and e.t.c.
  bool EnableDevice(int32_t id, bool a_enable) override { return true; }

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
    VkSampler textureSampler;
    uint32_t mipLevels;

    template<typename T>
    void createTextureImage(uint32_t width, uint32_t height, const T* image, VkFormat format);
    void createTextureImageView(VkFormat format);
    void createTextureSampler();
  public:
    template<typename T>
    Texture(VkDevice dev, uint32_t width, uint32_t height, const T* image) : deviceRef(dev) {
      createTextureImage(width, height, image, VkFormat(TypeToFormat<T>::format));
      createTextureImageView(VkFormat(TypeToFormat<T>::format));
      createTextureSampler();
    }

    ~Texture();

    VkDescriptorImageInfo getDescriptor() const;
  };

  struct Material {
    int textureIdx = -1;
    HydraLiteMath::float3 color;
  };

  class Mesh {
    VkBuffer vertexBuffer = {};
    VkBuffer indexBuffer = {};
    uint32_t indicesOffset = 0, indicesCount = 0;
    int materialId = -1;

  public:
    Mesh() {}
    Mesh(uint32_t indicesOffset, uint32_t indicesCount) :
      indicesOffset(indicesOffset),
      indicesCount(indicesCount)
    {}

    void bind(VkCommandBuffer command_buffer) const;

    void setBuffers(VkBuffer vertex_buffer, VkBuffer index_buffer) {
      vertexBuffer = vertex_buffer;
      indexBuffer = index_buffer;
    }

    uint32_t getIndicesCount() const {
      return indicesCount;
    }

    void setMaterialId(int id) {
      materialId = id;
    }

    int getMaterialId() const {
      return materialId;
    }
  };

  class HydraMesh {
    VkBuffer vertexBuffer;
    VkDeviceMemory vertexBufferMemory;
    VkBuffer indexBuffer;
    VkDeviceMemory indexBufferMemory;
    VkDevice device;

    std::vector<Mesh> meshes;

    void createVertexBuffer(const std::vector<Vertex>& vertices);
    template <typename T>
    void createIndexBuffer(const std::vector<T>& indices) {
      VkDeviceSize bufferSize = sizeof(indices[0]) * indices.size();

      VkBuffer stagingBuffer;
      VkDeviceMemory stagingBufferMemory;
      BufferManager::get().createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

      void* data;
      vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
      memcpy(data, indices.data(), (size_t)bufferSize);
      vkUnmapMemory(device, stagingBufferMemory);

      BufferManager::get().createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, indexBuffer, indexBufferMemory);

      BufferManager::get().copyBuffer(stagingBuffer, indexBuffer, bufferSize);

      vkDestroyBuffer(device, stagingBuffer, nullptr);
      vkFreeMemory(device, stagingBufferMemory, nullptr);
    }

  public:
    template <typename T>
    HydraMesh(VkDevice dev, const std::vector<Vertex>& vertices, const std::vector<T>& indices) : device(dev) {
      createVertexBuffer(vertices);
      createIndexBuffer(indices);
    }

    ~HydraMesh() {
      vkDestroyBuffer(device, indexBuffer, nullptr);
      vkFreeMemory(device, indexBufferMemory, nullptr);

      vkDestroyBuffer(device, vertexBuffer, nullptr);
      vkFreeMemory(device, vertexBufferMemory, nullptr);
    }

    void addMesh(Mesh mesh) {
      meshes.emplace_back(std::move(mesh));
      meshes.back().setBuffers(vertexBuffer, indexBuffer);
    }

    const Mesh& getMesh(int i) const {
      return meshes[i];
    }

    uint32_t getMeshesCount() const {
      return static_cast<uint32_t>(meshes.size());
    }
  };


  class InstancesCollection {
    std::vector<UniformBufferObject> ubos;
    int meshId;
    VkDescriptorSetLayout descriptorSetLayout;
    std::vector<VkBuffer> uniformBuffers;
    std::vector<VkDeviceMemory> uniformBuffersMemory;
    VkDescriptorPool descriptorPool;
    std::vector<VkDescriptorSet> descriptorSets;
    VkDevice device;
    int swapchain_images = 0;
    const int MAX_INSTANCES = 128;
    HydraLiteMath::float4 matColor;

    Texture *texture;

    void createUniformBuffers();
    void createDescriptorPool();
    void createDescriptorSets();

  public:
    InstancesCollection(VkDevice dev, VkDescriptorSetLayout layout, Texture *tex, int mesh_id, int swapchain_im, HydraLiteMath::float4 color)
      : device(dev), descriptorSetLayout(layout), texture(tex),
      meshId(mesh_id), swapchain_images(swapchain_im), matColor(color) {
      createUniformBuffers();
      createDescriptorPool();
      createDescriptorSets();
    }

    ~InstancesCollection();

    int getMeshId() const { return meshId; }
    bool instancesUpdated(const std::vector<HydraLiteMath::float4x4>& models) const;
    void updateInstances(const std::vector<HydraLiteMath::float4x4>& models);
    void bind(VkCommandBuffer command_buffer, VkPipelineLayout layout, int idx);
    void updateUniformBuffer(uint32_t current_image, const HydraLiteMath::float4x4& view, const HydraLiteMath::float4x4& proj);
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
    void transitionImageLayout(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout, uint32_t mipLevels);
    void copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height);
    void generateMipmaps(VkImage image, VkFormat imageFormat, int32_t texWidth, int32_t texHeight, uint32_t mipLevels);

    VkImageView createImageView(VkImage image, VkFormat format, VkImageAspectFlags aspectFlags, uint32_t mipLevels);
    void createImage(uint32_t width, uint32_t height, uint32_t mip_levels, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage,
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

  struct PipelineConfig {
    std::string vertexShaderPath, pixelShaderPath;
    bool hasVertexBuffer = false;
    VkRenderPass renderPass = {};
    VkDescriptorSetLayout descriptorSetLayout = {};
    uint32_t rtCount = 1;
  };

  struct DirectLightTemplate {
    HydraLiteMath::float3 color;
    float innerRadius; float outerRadius;
  };

  struct DirectLight {
    HydraLiteMath::float3 position;
    float innerRadius;
    HydraLiteMath::float3 direction;
    float outerRadius;
    HydraLiteMath::float3 color;
    float padding;
  };

  void createInstance();
  void pickPhysicalDevice();
  void createLogicalDevice();
  void createSurface();
  void createSwapChain();
  void createImageViews();
  void createPipelines();
  VkPipeline createGraphicsPipeline(const PipelineConfig&, VkPipelineLayout& layout);
  void createGbufferRenderPass();
  void createResolveRenderPass();
  void createFramebuffers();
  void createCommandPool();
  void createCommandBuffers();
  void createSyncObjects();
  void recreateSwapChain();
  void cleanupSwapChain();
  void createDefaultTexture();
  void createDepthResources();
  void createColorResources();
  void createColorSampler();
  void createBuffers();
  VkFormat findDepthFormat();
  QueueFamilyIndices GetQueueFamilyIndex();

  void createDescriptorSetLayout();
  void createDescriptorSets();
  void createDescriptorPool();
  void updateUniformBuffer(uint32_t current_image);

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
  VkFormat swapChainImageFormat;
  VkExtent2D swapChainExtent;
  VkDescriptorSetLayout gbufferDescriptorSetLayout;
  VkDescriptorSetLayout descriptorSetLayout;
  VkPipelineLayout gbufferPipelineLayout;
  VkPipelineLayout pipelineLayout;
  VkRenderPass gbufferRenderPass;
  VkRenderPass resolveRenderPass;
  VkPipeline graphicsPipeline;
  VkPipeline resolvePipeline;
  VkCommandPool commandPool;

  VkImage depthImage;
  VkDeviceMemory depthImageMemory;
  VkImageView depthImageView;
  VkSampler depthImageSampler;

  VkImage colorImage;
  VkDeviceMemory colorImageMemory;
  VkImageView colorImageView;
  VkSampler colorImageSampler;

  VkImage normalImage;
  VkDeviceMemory normalImageMemory;
  VkImageView normalImageView;
  VkSampler normalImageSampler;

  VkDescriptorPool descriptorPool;
  std::vector<VkDescriptorSet> descriptorSets;

  size_t currentFrame = 0;
  bool inited = false;
  std::map<int, std::unique_ptr<HydraMesh>> meshes;
  std::vector<std::vector<std::unique_ptr<InstancesCollection>>> instances;
  std::vector<std::unique_ptr<Texture>> textures;
  std::unique_ptr<Texture> defaultTexture;
  std::vector<Material> materials;
  std::unordered_map<uint32_t, DirectLightTemplate> directLightLib;
  std::vector<DirectLight> directLights;
  VkBuffer resolveConstants;
  VkDeviceMemory resolveConstantsMemory;

  VkBuffer lightsBuffer;
  VkDeviceMemory lightsBufferMemory;
};
