//	--------------------------------------------------------------------
//	Optimized implementation of simplex noise.
//	Based on stegu's simplex noise: https://github.com/stegu/webgl-noise.
//	Contact : atyuwen@gmail.com
//	Author : Yuwen Wu (https://atyuwen.github.io/)
//	License : Distributed under the MIT License.
//	--------------------------------------------------------------------

// Permuted congruential generator (only top 16 bits are well shuffled).
// References: 1. Mark Jarzynski and Marc Olano, "Hash Functions for GPU Rendering".
//             2. UnrealEngine/Random.ush. https://github.com/EpicGames/UnrealEngine
uint pcg3d16(uint3 p)
{
	uint3 v = p * 1664525u + 1013904223u;
	v.x += v.y*v.z; v.y += v.z*v.x; v.z += v.x*v.y;
	v.x += v.y*v.z;
	return v.x;
}
uint pcg4d16(uint4 p)
{
	uint4 v = p * 1664525u + 1013904223u;
	v.x += v.y*v.w; v.y += v.z*v.x; v.z += v.x*v.y; v.w += v.y*v.z;
	v.x += v.y*v.w;
	return v.x;
}

// Get random gradient from hash value.
float3 gradient3d(uint hash)
{
	float3 g = float3(hash.xxx & uint3(0x80000, 0x40000, 0x20000));
	return g * float3(1.0 / 0x40000, 1.0 / 0x20000, 1.0 / 0x10000) - 1.0;
}
float4 gradient4d(uint hash)
{
	float4 g = float4(hash.xxxx & uint4(0x80000, 0x40000, 0x20000, 0x10000));
	return g * float4(1.0 / 0x40000, 1.0 / 0x20000, 1.0 / 0x10000, 1.0 / 0x8000) - 1.0;
}

// 3D Simplex Noise. Assume p is in the range [-32768, 32767].
float SimplexNoise3D(float3 p)
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
	uint hash0 = pcg3d16((uint3)i);
	uint hash1 = pcg3d16((uint3)(i + i1));
	uint hash2 = pcg3d16((uint3)(i + i2));
	uint hash3 = pcg3d16((uint3)(i + 1 ));

	float3 p0 = gradient3d(hash0);
	float3 p1 = gradient3d(hash1);
	float3 p2 = gradient3d(hash2);
	float3 p3 = gradient3d(hash3);

	// Mix final noise value.
	float4 m = saturate(0.5 - float4(dot(x0, x0), dot(x1, x1), dot(x2, x2), dot(x3, x3)));
	float4 mt = m * m;
	float4 m4 = mt * mt;
	return 62.6 * dot(m4, float4(dot(x0, p0), dot(x1, p1), dot(x2, p2), dot(x3, p3)));
}

// 4D Simplex Noise. Assume p is in the range [-32768, 32767].
// Note the quality is worse than stegu's implementation.
float3 SimplexNoise4D(float4 p)
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
	uint hash0 = pcg4d16((uint4)i);
	uint hash1 = pcg4d16((uint4)(i + i1));
	uint hash2 = pcg4d16((uint4)(i + i2));
	uint hash3 = pcg4d16((uint4)(i + i3));
	uint hash4 = pcg4d16((uint4)(i + 1 ));

	float4 p0 = gradient4d(hash0);
	float4 p1 = gradient4d(hash1);
	float4 p2 = gradient4d(hash2);
	float4 p3 = gradient4d(hash3);
	float4 p4 = gradient4d(hash4);

	// Mix contributions from the five corners
	float3 m0 = saturate(0.5 - float3(dot(x0,x0), dot(x1,x1), dot(x2,x2)));
	float2 m1 = saturate(0.5 - float2(dot(x3,x3), dot(x4,x4)            ));
	float3 m0t = m0 * m0;
	float2 m1t = m1 * m1;
	float3 m04 = m0t * m0t;
	float2 m14 = m1t * m1t;
	return (dot(m04, float3(dot(p0, x0), dot(p1, x1), dot(p2, x2)))
	      + dot(m14, float2(dot(p3, x3), dot(p4, x4)))) * 54.7;
}
