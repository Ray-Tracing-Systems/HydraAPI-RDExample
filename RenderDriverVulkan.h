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

  void setView(const HydraLiteMath::float3 eye, const HydraLiteMath::float3 center, const HydraLiteMath::float3 up) {
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
    uint32_t graphicsFamily;
    uint32_t presentFamily;
  };

  class Mesh {
    VkBuffer vertexBuffer;
    VkDeviceMemory vertexBufferMemory;
    VkBuffer indexBuffer;
    VkDeviceMemory indexBufferMemory;
    VkDevice device;
    uint32_t indicesCount;

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
    Mesh(VkDevice dev) : device(dev) {}

    ~Mesh() {
      vkDestroyBuffer(device, indexBuffer, nullptr);
      vkFreeMemory(device, indexBufferMemory, nullptr);

      vkDestroyBuffer(device, vertexBuffer, nullptr);
      vkFreeMemory(device, vertexBufferMemory, nullptr);
    }

    template <typename T>
    void createMesh(const std::vector<Vertex>& vertices, const std::vector<T>& indices) {
      createVertexBuffer(vertices);
      createIndexBuffer(indices);
      indicesCount = static_cast<uint32_t>(indices.size());
    }

    void bind(VkCommandBuffer command_buffer);

    uint32_t getIndicesCount() const {
      return indicesCount;
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

    void createUniformBuffers();
    void createDescriptorPool();
    void createDescriptorSets();

  public:
    InstancesCollection(VkDevice dev, VkDescriptorSetLayout layout, int mesh_id, int swapchain_im)
      : device(dev), descriptorSetLayout(layout), meshId(mesh_id), swapchain_images(swapchain_im) {
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
  private:
    BufferManager() {}
    ~BufferManager() {}
  };

  void createInstance();
  void pickPhysicalDevice();
  void createLogicalDevice();
  void createSurface();
  void createSwapChain();
  void createImageViews();
  void createGraphicsPipeline();
  void createRenderPass();
  void createFramebuffers();
  void createCommandPool();
  void createCommandBuffers();
  void createSyncObjects();
  void recreateSwapChain();
  void cleanupSwapChain();
  QueueFamilyIndices GetQueueFamilyIndex(VkPhysicalDevice physicalDevice);

  void createDescriptorSetLayout();
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
  VkQueue presentQueue;
  std::vector<VkCommandBuffer> commandBuffers;
  std::vector<VkImage> swapChainImages;
  std::vector<VkImageView> swapChainImageViews;
  std::vector<VkFramebuffer> swapChainFramebuffers;
  VkFormat swapChainImageFormat;
  VkExtent2D swapChainExtent;
  VkDescriptorSetLayout descriptorSetLayout;
  VkPipelineLayout pipelineLayout;
  VkRenderPass renderPass;
  VkPipeline graphicsPipeline;
  VkCommandPool commandPool;
  size_t currentFrame = 0;
  std::map<int, std::unique_ptr<Mesh>> meshes;
  std::vector<std::unique_ptr<InstancesCollection>> instances;
};


struct RD_Vulkan_Debug : public RD_Vulkan
{
  RD_Vulkan_Debug()
  {
    m_renderNormalLength = 0.5f;
    m_drawNormals        = true;
    m_drawTangents       = false;
    m_drawSolid          = true;
    m_drawWire           = false;
    m_drawAxis           = false;

    m_axisArrorLen       = 1.0f;
    m_axisArrorThickness = 0.1f;
    m_meshNum            = 0;
  }

  void GetRenderDriverName(std::wstring &name) override { name = std::wstring(L"vulkanDebug");};

  void ClearAll() override;
  HRDriverAllocInfo AllocAll(HRDriverAllocInfo a_info) override;

  bool UpdateMesh(int32_t a_meshId, pugi::xml_node a_meshNode, const HRMeshDriverInput& a_input,
      const HRBatchInfo* a_batchList, int32_t listSize) override;

  bool UpdateSettings(pugi::xml_node a_settingsNode) override;

  /////////////////////////////////////////////////////////////////////////////////////////////

  void InstanceMeshes(int32_t a_mesh_id, const float* a_matrices, int32_t a_instNum, const int* a_lightInstId,
      const int* a_remapId, const int* a_realInstId) override;

  void EndScene() override;

protected:
  unsigned int m_meshNum;
  float m_renderNormalLength;

  bool m_drawSolid;
  bool m_drawWire;
  bool m_drawAxis;
  bool m_drawNormals;
  bool m_drawTangents;

  float m_axisArrorLen;
  float m_axisArrorThickness;
};

struct RD_Vulkan_ShowCustomAttr : public RD_Vulkan
{
  RD_Vulkan_ShowCustomAttr() = default;

  void GetRenderDriverName(std::wstring &name) override { name = std::wstring(L"vulkanTestCustomAttributes");};

  bool UpdateMesh(int32_t a_meshId, pugi::xml_node a_meshNode, const HRMeshDriverInput& a_input,
      const HRBatchInfo* a_batchList, int32_t listSize) override;

protected:
  
};


