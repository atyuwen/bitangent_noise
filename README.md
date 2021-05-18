# Bitangent Noise

[Curl Noise](https://www.cct.lsu.edu/~fharhad/ganbatte/siggraph2007/CD2/content/papers/046-bridson.pdf) by Robert Bridson is a widely known method that can generate divergence-free noise. This divergence-free property makes it extremely suitable for driving particles to move like real fluid motion.

Here is another divergence-free noise generator that is **computationally cheaper** than curl noise. I thought it was new and named it **Bitangent Noise**, but later I found it was already invented by [Ivan DeWolf](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.93.7627&rep=rep1&type=pdf) in 2005. (I'm wondering why it is so less popular comparing to curl noise.)

The implementation is carefully optimized, and both HLSL and GLSL codes are provided for your convenience. More details can be found [here](https://atyuwen.github.io/posts/bitangent-noise/). Following image shows a particle system that is updated using bitangent noise.

![image](image.png?raw=true)

## Performance

These performance data are measured on a Nvidia GTX 1060 card, where each noise function is executed 1280 * 720 * 10 times.

| Noise Function           |   Cost      | Desc                                                                                                                                |
|--------------------------|  ---------  |-------------------------------------------------------------------------------------------------------------------------------------|
| snoise3d                 |   1530 μs   | [stegu's 3d simplex nosie](https://github.com/stegu/webgl-noise/blob/master/src/noise3D.glsl)                                       |
| **SimplexNoise3D**       | **1153 μs** | [optimized 3d simplex noise](https://github.com/atyuwen/bitangent_noise/blob/main/Develop/SimplexNoise.hlsl#L41)                    |
| snoise4d                 |   2578 μs   | [stegu's 4d simplex nosie](https://github.com/stegu/webgl-noise/blob/master/src/noise4D.glsl)                                       |
| **SimplexNoise4D**       | **1798 μs** | [optimized 4d simplex noise](https://github.com/atyuwen/bitangent_noise/blob/main/Develop/SimplexNoise.hlsl#L84)                    |
| BitangentNoise3D_ref     |   2991 μs   | [3d bitangent noise, reference version ](https://github.com/atyuwen/bitangent_noise/blob/main/Develop/BitangentNoise_ref.hlsl#L219) |
| **BitangentNoise3D**     | **1534 μs** | [optimized 3d bitangent noise](https://github.com/atyuwen/bitangent_noise/blob/main/BitangentNoise.hlsl#L41)                        |
| BitangentNoise4D_ref     |   4365 μs   | [4d bitangent noise, reference version](https://github.com/atyuwen/bitangent_noise/blob/main/Develop/BitangentNoise_ref.hlsl#L227)  |
| BitangentNoise4DFast_ref |   3152 μs   | [4d bitangent noise, low quality](https://github.com/atyuwen/bitangent_noise/blob/main/Develop/BitangentNoise_ref.hlsl#L239)        |
| **BitangentNoise4D**     | **2413 μs** | [optimized 4d bitangent noise](https://github.com/atyuwen/bitangent_noise/blob/main/BitangentNoise.hlsl#L97)                        |