#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(binding = 0) uniform sampler2D resolvedScene;

layout(location = 0) in vec2 fragTexCoord;

layout(location = 0) out vec4 outColor;

float luminance(vec3 color) {
    return dot(color, vec3(0.2126, 0.7152, 0.0722));
}

void main() {
    vec3 avgColor = textureLod(resolvedScene, vec2(0.5, 0.5), 100).rgb;
    float avgLum = luminance(avgColor);
    vec3 texelColor = texture(resolvedScene, fragTexCoord).rgb;
    outColor.rgb = 1 - exp(-texelColor / avgLum);
}
