//	--------------------------------------------------------------------
//	Optimized implementation of 3D/4D bitangent noise.
//	Based on stegu's simplex noise: https://github.com/stegu/webgl-noise.
//	Contact : atyuwen@gmail.com
//	Author : Yuwen Wu (https://atyuwen.github.io/)
//	License : Distributed under the MIT License.
//	--------------------------------------------------------------------

// Permuted congruential generator (only top 16 bits are well shuffled).
// References: 1. Mark Jarzynski and Marc Olano, "Hash Functions for GPU Rendering".
//             2. UnrealEngine/Random.ush. https://github.com/EpicGames/UnrealEngine
uint2 _pcg3d16(uint3 p)
{
	uint3 v = p * 1664525u + 1013904223u;
	v.x += v.y*v.z; v.y += v.z*v.x; v.z += v.x*v.y;
	v.x += v.y*v.z; v.y += v.z*v.x;
	return v.xy;
}
uint2 _pcg4d16(uint4 p)
{
	uint4 v = p * 1664525u + 1013904223u;
	v.x += v.y*v.w; v.y += v.z*v.x; v.z += v.x*v.y; v.w += v.y*v.z;
	v.x += v.y*v.w; v.y += v.z*v.x;
	return v.xy;
}

// Get random gradient from hash value.
float3 _gradient3d(uint hash)
{
	float3 g = float3(hash.xxx & uint3(0x80000, 0x40000, 0x20000));
	return g * float3(1.0 / 0x40000, 1.0 / 0x20000, 1.0 / 0x10000) - 1.0;
}
float4 _gradient4d(uint hash)
{
	float4 g = float4(hash.xxxx & uint4(0x80000, 0x40000, 0x20000, 0x10000));
	return g * float4(1.0 / 0x40000, 1.0 / 0x20000, 1.0 / 0x10000, 1.0 / 0x8000) - 1.0;
}

// Optimized 3D Bitangent Noise. Approximately 113 instruction slots used.
// Assume p is in the range [-32768, 32767].
float3 BitangentNoise3D(float3 p)
{
	const float2 C = float2(1.0 / 6.0, 1.0 / 3.0);
	const float4 D = float4(0.0, 0.5, 1.0, 2.0);

	// First corner
	float3 i = floor(p + dot(p, C.yyy));
	float3 x0 = p - i + dot(i, C.xxx);

	// Other corners
	float3 g = step(x0.yzx, x0.xyz);
	float3 l = 1.0 - g;
	float3 i1 = min(g.xyz, l.zxy);
	float3 i2 = max(g.xyz, l.zxy);

	// x0 = x0 - 0.0 + 0.0 * C.xxx;
	// x1 = x0 - i1  + 1.0 * C.xxx;
	// x2 = x0 - i2  + 2.0 * C.xxx;
	// x3 = x0 - 1.0 + 3.0 * C.xxx;
	float3 x1 = x0 - i1 + C.xxx;
	float3 x2 = x0 - i2 + C.yyy; // 2.0*C.x = 1/3 = C.y
	float3 x3 = x0 - D.yyy;      // -1.0+3.0*C.x = -0.5 = -D.y

	i = i + 32768.5;
	uint2 hash0 = _pcg3d16((uint3)i);
	uint2 hash1 = _pcg3d16((uint3)(i + i1));
	uint2 hash2 = _pcg3d16((uint3)(i + i2));
	uint2 hash3 = _pcg3d16((uint3)(i + 1 ));

	float3 p00 = _gradient3d(hash0.x); float3 p01 = _gradient3d(hash0.y);
	float3 p10 = _gradient3d(hash1.x); float3 p11 = _gradient3d(hash1.y);
	float3 p20 = _gradient3d(hash2.x); float3 p21 = _gradient3d(hash2.y);
	float3 p30 = _gradient3d(hash3.x); float3 p31 = _gradient3d(hash3.y);

	// Calculate noise gradients.
	float4 m = saturate(0.5 - float4(dot(x0, x0), dot(x1, x1), dot(x2, x2), dot(x3, x3)));
	float4 mt = m * m;
	float4 m4 = mt * mt;

	mt = mt * m;
	float4 pdotx = float4(dot(p00, x0), dot(p10, x1), dot(p20, x2), dot(p30, x3));
	float4 temp = mt * pdotx;
	float3 gradient0 = -8.0 * (temp.x * x0 + temp.y * x1 + temp.z * x2 + temp.w * x3);
	gradient0 += m4.x * p00 + m4.y * p10 + m4.z * p20 + m4.w * p30;

	pdotx = float4(dot(p01, x0), dot(p11, x1), dot(p21, x2), dot(p31, x3));
	temp = mt * pdotx;
	float3 gradient1 = -8.0 * (temp.x * x0 + temp.y * x1 + temp.z * x2 + temp.w * x3);
	gradient1 += m4.x * p01 + m4.y * p11 + m4.z * p21 + m4.w * p31;

	// The cross products of two gradients is divergence free.
	return cross(gradient0, gradient1) * 3918.76;
}

// 4D Bitangent noise. Approximately 163 instruction slots used.
// Assume p is in the range [-32768, 32767].
float3 BitangentNoise4D(float4 p)
{
	const float4 F4 = 0.309016994374947451;
	const float4  C = float4( 0.138196601125011,  // (5 - sqrt(5))/20  G4
	                          0.276393202250021,  // 2 * G4
	                          0.414589803375032,  // 3 * G4
	                         -0.447213595499958); // -1 + 4 * G4

	// First corner
	float4 i  = floor(p + dot(p, F4) );
	float4 x0 = p -   i + dot(i, C.xxxx);

	// Other corners

	// Rank sorting originally contributed by Bill Licea-Kane, AMD (formerly ATI)
	float4 i0;
	float3 isX = step( x0.yzw, x0.xxx );
	float3 isYZ = step( x0.zww, x0.yyz );
	// i0.x = dot( isX, float3( 1.0 ) );
	i0.x = isX.x + isX.y + isX.z;
	i0.yzw = 1.0 - isX;
	// i0.y += dot( isYZ.xy, float2( 1.0 ) );
	i0.y += isYZ.x + isYZ.y;
	i0.zw += 1.0 - isYZ.xy;
	i0.z += isYZ.z;
	i0.w += 1.0 - isYZ.z;

	// i0 now contains the unique values 0,1,2,3 in each channel
	float4 i3 = saturate( i0 );
	float4 i2 = saturate( i0 - 1.0 );
	float4 i1 = saturate( i0 - 2.0 );

	// x0 = x0 - 0.0 + 0.0 * C.xxxx
	// x1 = x0 - i1  + 1.0 * C.xxxx
	// x2 = x0 - i2  + 2.0 * C.xxxx
	// x3 = x0 - i3  + 3.0 * C.xxxx
	// x4 = x0 - 1.0 + 4.0 * C.xxxx
	float4 x1 = x0 - i1 + C.xxxx;
	float4 x2 = x0 - i2 + C.yyyy;
	float4 x3 = x0 - i3 + C.zzzz;
	float4 x4 = x0 + C.wwww;

	i = i + 32768.5;
	uint2 hash0 = _pcg4d16((uint4)i);
	uint2 hash1 = _pcg4d16((uint4)(i + i1));
	uint2 hash2 = _pcg4d16((uint4)(i + i2));
	uint2 hash3 = _pcg4d16((uint4)(i + i3));
	uint2 hash4 = _pcg4d16((uint4)(i + 1 ));

	float4 p00 = _gradient4d(hash0.x); float4 p01 = _gradient4d(hash0.y);
	float4 p10 = _gradient4d(hash1.x); float4 p11 = _gradient4d(hash1.y);
	float4 p20 = _gradient4d(hash2.x); float4 p21 = _gradient4d(hash2.y);
	float4 p30 = _gradient4d(hash3.x); float4 p31 = _gradient4d(hash3.y);
	float4 p40 = _gradient4d(hash4.x); float4 p41 = _gradient4d(hash4.y);

	// Calculate noise gradients.
	float3 m0 = saturate(0.6 - float3(dot(x0, x0), dot(x1, x1), dot(x2, x2)));
	float2 m1 = saturate(0.6 - float2(dot(x3, x3), dot(x4, x4)             ));
	float3 m02 = m0 * m0; float3 m03 = m02 * m0;
	float2 m12 = m1 * m1; float2 m13 = m12 * m1;

	float3 temp0 = m02 * float3(dot(p00, x0), dot(p10, x1), dot(p20, x2));
	float2 temp1 = m12 * float2(dot(p30, x3), dot(p40, x4));
	float4 grad0 = -6.0 * (temp0.x * x0 + temp0.y * x1 + temp0.z * x2 + temp1.x * x3 + temp1.y * x4);
	grad0 += m03.x * p00 + m03.y * p10 + m03.z * p20 + m13.x * p30 + m13.y * p40;

	temp0 = m02 * float3(dot(p01, x0), dot(p11, x1), dot(p21, x2));
	temp1 = m12 * float2(dot(p31, x3), dot(p41, x4));
	float4 grad1 = -6.0 * (temp0.x * x0 + temp0.y * x1 + temp0.z * x2 + temp1.x * x3 + temp1.y * x4);
	grad1 += m03.x * p01 + m03.y * p11 + m03.z * p21 + m13.x * p31 + m13.y * p41;

	// The cross products of two gradients is divergence free.
	return cross(grad0.xyz, grad1.xyz) * 81.0;
}
