//	--------------------------------------------------------------------
//	Reference implementation of 3D/4D bitangent noise.
//	Based on stegu's simplex noise: https://github.com/stegu/webgl-noise.
//	Contact : atyuwen@gmail.com
//	Author : Yuwen Wu (atyuwen)
//	License : Distributed under the MIT License.
//	--------------------------------------------------------------------

float mod289(float x)
{
	return x - floor(x * (1.0 / 289.0)) * 289.0;
}
float3 mod289(float3 x)
{
	return x - floor(x * (1.0 / 289.0)) * 289.0;
}
float4 mod289(float4 x)
{
	return x - floor(x * (1.0 / 289.0)) * 289.0;
}

float permute(float x)
{
	return mod289(((x*34.0) + 1.0)*x);
}
float4 permute(float4 x)
{
	return mod289(((x*34.0) + 1.0)*x);
}

float taylorInvSqrt(float r)
{
	return 1.79284291400159 - 0.85373472095314 * r;
}
float4 taylorInvSqrt(float4 r)
{
	return 1.79284291400159 - 0.85373472095314 * r;
}

float3 SimplexNoise3DGrad(float3 v)
{
	const float2 C = float2(1.0 / 6.0, 1.0 / 3.0);
	const float4 D = float4(0.0, 0.5, 1.0, 2.0);

	// First corner
	float3 i = floor(v + dot(v, C.yyy));
	float3 x0 = v - i + dot(i, C.xxx);

	// Other corners
	float3 g = step(x0.yzx, x0.xyz);
	float3 l = 1.0 - g;
	float3 i1 = min(g.xyz, l.zxy);
	float3 i2 = max(g.xyz, l.zxy);

	//   x0 = x0 - 0.0 + 0.0 * C.xxx;
	//   x1 = x0 - i1  + 1.0 * C.xxx;
	//   x2 = x0 - i2  + 2.0 * C.xxx;
	//   x3 = x0 - 1.0 + 3.0 * C.xxx;
	float3 x1 = x0 - i1 + C.xxx;
	float3 x2 = x0 - i2 + C.yyy; // 2.0*C.x = 1/3 = C.y
	float3 x3 = x0 - D.yyy;      // -1.0+3.0*C.x = -0.5 = -D.y

	// Permutations
	i = mod289(i);
	float4 p = permute(permute(permute(
		i.z + float4(0.0, i1.z, i2.z, 1.0))
		+ i.y + float4(0.0, i1.y, i2.y, 1.0))
		+ i.x + float4(0.0, i1.x, i2.x, 1.0));

	// Gradients: 7x7 points over a square, mapped onto an octahedron.
	// The ring size 17*17 = 289 is close to a multiple of 49 (49*6 = 294)
	float n_ = 0.142857142857; // 1.0/7.0
	float3 ns = n_ * D.wyz - D.xzx;

	float4 j = p - 49.0 * floor(p * ns.z * ns.z);  //  mod(p,7*7)

	float4 x_ = floor(j * ns.z);
	float4 y_ = floor(j - 7.0 * x_);    // mod(j,N)

	float4 x = x_ *ns.x + ns.yyyy;
	float4 y = y_ *ns.x + ns.yyyy;
	float4 h = 1.0 - abs(x) - abs(y);

	float4 b0 = float4(x.xy, y.xy);
	float4 b1 = float4(x.zw, y.zw);

	//float4 s0 = float4(lessThan(b0,0.0))*2.0 - 1.0;
	//float4 s1 = float4(lessThan(b1,0.0))*2.0 - 1.0;
	float4 s0 = floor(b0)*2.0 + 1.0;
	float4 s1 = floor(b1)*2.0 + 1.0;
	float4 sh = -step(h, float4(0, 0, 0, 0));

	float4 a0 = b0.xzyw + s0.xzyw*sh.xxyy;
	float4 a1 = b1.xzyw + s1.xzyw*sh.zzww;

	float3 p0 = float3(a0.xy, h.x);
	float3 p1 = float3(a0.zw, h.y);
	float3 p2 = float3(a1.xy, h.z);
	float3 p3 = float3(a1.zw, h.w);

	// Normalise gradients
	float4 norm = taylorInvSqrt(float4(dot(p0, p0), dot(p1, p1), dot(p2, p2), dot(p3, p3)));
	p0 *= norm.x;
	p1 *= norm.y;
	p2 *= norm.z;
	p3 *= norm.w;

	// lerp final noise value
	float4 m = max(0.6 - float4(dot(x0, x0), dot(x1, x1), dot(x2, x2), dot(x3, x3)), 0.0);
	float4 m2 = m * m;
	float4 m4 = m2 * m2;
	float4 pdotx = float4(dot(p0, x0), dot(p1, x1), dot(p2, x2), dot(p3, x3));

	// Determine noise gradient
	float4 temp = m2 * m * pdotx;
	float3 gradient = -8.0 * (temp.x * x0 + temp.y * x1 + temp.z * x2 + temp.w * x3);
	gradient += m4.x * p0 + m4.y * p1 + m4.z * p2 + m4.w * p3;
	gradient *= 42.0;
	return gradient;
}

float4 grad4(float j, float4 ip)
{
	const float4 ones = float4(1.0, 1.0, 1.0, -1.0);
	float4 p,s;
	p.xyz = floor( frac(float3(j, j, j) * ip.xyz) * 7.0) * ip.z - 1.0;
	p.w = 1.5 - dot(abs(p.xyz), ones.xyz);
	s = float4(p < float4(0, 0, 0, 0));
	p.xyz = p.xyz + (s.xyz*2.0 - 1.0) * s.www;
	return p;
}

float4 SimplexNoise4DGrad(float4 v)
{
	const float4  C = float4( 0.138196601125011,  // (5 - sqrt(5))/20  G4
	                          0.276393202250021,  // 2 * G4
	                          0.414589803375032,  // 3 * G4
	                         -0.447213595499958); // -1 + 4 * G4

	// First corner
	float4 F4 = 0.309016994374947451;
	float4 i  = floor(v + dot(v, F4) );
	float4 x0 = v -   i + dot(i, C.xxxx);

	// Other corners

	// Rank sorting originally contributed by Bill Licea-Kane, AMD (formerly ATI)
	float4 i0;
	float3 isX = step( x0.yzw, x0.xxx );
	float3 isYZ = step( x0.zww, x0.yyz );
	//  i0.x = dot( isX, float3( 1.0 ) );
	i0.x = isX.x + isX.y + isX.z;
	i0.yzw = 1.0 - isX;
	//  i0.y += dot( isYZ.xy, float2( 1.0 ) );
	i0.y += isYZ.x + isYZ.y;
	i0.zw += 1.0 - isYZ.xy;
	i0.z += isYZ.z;
	i0.w += 1.0 - isYZ.z;

	// i0 now contains the unique values 0,1,2,3 in each channel
	float4 i3 = clamp( i0, 0.0, 1.0 );
	float4 i2 = clamp( i0-1.0, 0.0, 1.0 );
	float4 i1 = clamp( i0-2.0, 0.0, 1.0 );

	//  x0 = x0 - 0.0 + 0.0 * C.xxxx
	//  x1 = x0 - i1  + 1.0 * C.xxxx
	//  x2 = x0 - i2  + 2.0 * C.xxxx
	//  x3 = x0 - i3  + 3.0 * C.xxxx
	//  x4 = x0 - 1.0 + 4.0 * C.xxxx
	float4 x1 = x0 - i1 + C.xxxx;
	float4 x2 = x0 - i2 + C.yyyy;
	float4 x3 = x0 - i3 + C.zzzz;
	float4 x4 = x0 + C.wwww;

	// Permutations
	i = mod289(i); 
	float j0 = permute( permute( permute( permute(i.w) + i.z) + i.y) + i.x);
	float4 j1 = permute( permute( permute( permute (
	           i.w + float4(i1.w, i2.w, i3.w, 1.0 ))
	         + i.z + float4(i1.z, i2.z, i3.z, 1.0 ))
	         + i.y + float4(i1.y, i2.y, i3.y, 1.0 ))
	         + i.x + float4(i1.x, i2.x, i3.x, 1.0 ));

	// Gradients: 7x7x6 points over a cube, mapped onto a 4-cross polytope
	// 7*7*6 = 294, which is close to the ring size 17*17 = 289.
	float4 ip = float4(1.0/294.0, 1.0/49.0, 1.0/7.0, 0.0) ;

	float4 p0 = grad4(j0,   ip);
	float4 p1 = grad4(j1.x, ip);
	float4 p2 = grad4(j1.y, ip);
	float4 p3 = grad4(j1.z, ip);
	float4 p4 = grad4(j1.w, ip);

	// Normalise gradients
	float4 norm = taylorInvSqrt(float4(dot(p0,p0), dot(p1,p1), dot(p2, p2), dot(p3,p3)));
	p0 *= norm.x;
	p1 *= norm.y;
	p2 *= norm.z;
	p3 *= norm.w;
	p4 *= taylorInvSqrt(dot(p4,p4));

	// Mix contributions from the five corners
	float3 m0 = max(0.6 - float3(dot(x0,x0), dot(x1,x1), dot(x2,x2)), 0.0);
	float2 m1 = max(0.6 - float2(dot(x3,x3), dot(x4,x4)            ), 0.0);
	float3 m02 = m0 * m0;
	float2 m12 = m1 * m1;
	float3 m04 = m02 * m02;
	float2 m14 = m12 * m12;

	float3 temp0 = (m02 * m0) * float3( dot( p0, x0 ), dot( p1, x1 ), dot( p2, x2 ) );
	float2 temp1 = (m12 * m1) * float2( dot( p3, x3 ), dot( p4, x4 ) );
	float4 grad = -8.0 * (temp0.x * x0 + temp0.y * x1 + temp0.z * x2 + temp1.x * x3 + temp1.y * x4);
	grad += m04.x * p0 + m04.y * p1 + m04.z * p2 + m14.x * p3 + m14.y * p4;
	grad *= 49;

	return grad;
}

// 3D Bitangent noise. Approximately 223 instruction slots used.
float3 BitangentNoise3D(float3 p)
{
	float3 dA = SimplexNoise3DGrad(p);
	float3 dB = SimplexNoise3DGrad(p + float3(31.416, -47.853, 12.679));
	return cross(dA, dB);
}

// 4D Bitangent noise. Approximately 318 instruction slots used.
float3 BitangentNoise4D(float4 p)
{
	float3 dA = SimplexNoise4DGrad(p).xyz;
	float3 dB = SimplexNoise3DGrad(p.xyz + float3(31.416, -47.853, 12.679));
	return cross(dA, dB);
}
