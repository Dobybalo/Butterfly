//=============================================================================================
// Szamitogepes grafika hazi feladat keret. Ervenyes 2018-tol.
// A //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// sorokon beluli reszben celszeru garazdalkodni, mert a tobbit ugyis toroljuk.
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!! 
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Döbrössy Bálint	
// Neptun : HDWUML
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
#define _USE_MATH_DEFINES		// Van M_PI
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>

#if defined(__APPLE__)
#include <GLUT/GLUT.h>
#include <OpenGL/gl3.h>
#include <OpenGL/glu.h>
#else
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__)
#include <windows.h>
#endif
#include <GL/glew.h>		// must be downloaded 
#include <GL/freeglut.h>	// must be downloaded unless you have an Apple
#endif

const unsigned int windowWidth = 600, windowHeight = 600;

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// You are supposed to modify the code from here...

// OpenGL major and minor versions
int majorVersion = 3, minorVersion = 3;

void getErrorInfo(unsigned int handle) {
	int logLen;
	glGetShaderiv(handle, GL_INFO_LOG_LENGTH, &logLen);
	if (logLen > 0) {
		char * log = new char[logLen];
		int written;
		glGetShaderInfoLog(handle, logLen, &written, log);
		printf("Shader log:\n%s", log);
		delete log;
	}
}

// check if shader could be compiled
void checkShader(unsigned int shader, char * message) {
	int OK;
	glGetShaderiv(shader, GL_COMPILE_STATUS, &OK);
	if (!OK) { printf("%s!\n", message); getErrorInfo(shader); }
}

// check if shader could be linked
void checkLinking(unsigned int program) {
	int OK;
	glGetProgramiv(program, GL_LINK_STATUS, &OK);
	if (!OK) { printf("Failed to link shader program!\n"); getErrorInfo(program); }
}

// vertex shader in GLSL
const char * vertexSource = R"(
	#version 330
    precision highp float;

	uniform mat4 MVP;			// Model-View-Projection matrix in row-major format

	layout(location = 0) in vec2 vertexPosition;	// Attrib Array 0
	layout(location = 1) in vec3 vertexColor;	    // Attrib Array 1
	out vec3 color;									// output attribute

	void main() {
		color = vertexColor;														// copy color from input to output
		gl_Position = vec4(vertexPosition.x, vertexPosition.y, 0, 1) * MVP; 		// transform to clipping space
	}
)";

// fragment shader in GLSL
const char * fragmentSource = R"(
	#version 330
    precision highp float;

	in vec3 color;				// variable input: interpolated color of vertex shader
	out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation

	void main() {
		fragmentColor = vec4(color, 1); // extend RGB to RGBA
	}
)";

// row-major matrix 4x4
struct mat4 {
	float m[4][4];
public:
	mat4() {}
	mat4(float m00, float m01, float m02, float m03,
		float m10, float m11, float m12, float m13,
		float m20, float m21, float m22, float m23,
		float m30, float m31, float m32, float m33) {
		m[0][0] = m00; m[0][1] = m01; m[0][2] = m02; m[0][3] = m03;
		m[1][0] = m10; m[1][1] = m11; m[1][2] = m12; m[1][3] = m13;
		m[2][0] = m20; m[2][1] = m21; m[2][2] = m22; m[2][3] = m23;
		m[3][0] = m30; m[3][1] = m31; m[3][2] = m32; m[3][3] = m33;
	}

	mat4 operator*(const mat4& right) const {
		mat4 result;
		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				result.m[i][j] = 0;
				for (int k = 0; k < 4; k++) result.m[i][j] += m[i][k] * right.m[k][j];
			}
		}
		return result;
	}
	operator float*() { return &m[0][0]; }
};

// 3D point in homogeneous coordinates
struct vec4 {
	float x, y, z, w;
	vec4(float _x = 0, float _y = 0, float _z = 0, float _w = 1) { x = _x; y = _y; z = _z; w = _w; }

	vec4 operator*(const mat4& mat) const {
		return vec4(x * mat.m[0][0] + y * mat.m[1][0] + z * mat.m[2][0] + w * mat.m[3][0],
			x * mat.m[0][1] + y * mat.m[1][1] + z * mat.m[2][1] + w * mat.m[3][1],
			x * mat.m[0][2] + y * mat.m[1][2] + z * mat.m[2][2] + w * mat.m[3][2],
			x * mat.m[0][3] + y * mat.m[1][3] + z * mat.m[2][3] + w * mat.m[3][3]);
	}

	vec4 operator*(float f) {
		vec4 result(x * f, y * f);
		return result;
	}

	vec4 operator+(const vec4& other_vector) {
		vec4 result(x + other_vector.x, y + other_vector.y);
		return result;
	}
};

// 2D camera
struct Camera {
	float wCx, wCy;	// center in world coordinates
	float wWx, wWy;	// width and height in world coordinates
public:
	Camera() {
		Animate(0);
	}

	mat4 V() { // view matrix: translates the center to the origin
		return mat4(1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			-wCx, -wCy, 0, 1);
	}

	mat4 P() { // projection matrix: scales it to be a square of edge length 2
		return mat4(2 / wWx, 0, 0, 0,
			0, 2 / wWy, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1);
	}

	mat4 Vinv() { // inverse view matrix
		return mat4(1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			wCx, wCy, 0, 1);
	}

	mat4 Pinv() { // inverse projection matrix
		return mat4(wWx / 2, 0, 0, 0,
			0, wWy / 2, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1);
	}

	void Animate(float t) {
		wCx = 0; // 10 * cosf(t);
		wCy = 0;
		wWx = 20;
		wWy = 20;
	}
};

// 2D camera
Camera camera;

// handle of the shader program
unsigned int shaderProgram;

class LineStrip {

protected:
	GLuint vao, vbo;        // vertex array object, vertex buffer object
	float  vertexData[10000]; // interleaved data of coordinates and colors
	int    nVertices;       // number of vertices
public:

	LineStrip() {
		nVertices = 0;
	}
	void Create() {
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);

		glGenBuffers(1, &vbo); // Generate 1 vertex buffer object
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		// Enable the vertex attribute arrays
		glEnableVertexAttribArray(0);  // attribute array 0
		glEnableVertexAttribArray(1);  // attribute array 1
									   // Map attribute array 0 to the vertex data of the interleaved vbo
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), reinterpret_cast<void*>(0)); // attribute array, components/attribute, component type, normalize?, stride, offset
																										// Map attribute array 1 to the color data of the interleaved vbo
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), reinterpret_cast<void*>(2 * sizeof(float)));
	}

	void AddPoint(float cX, float cY) {
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		if (nVertices >= 2000) return;

		vec4 wVertex = vec4(cX, cY, 0, 1) * camera.Pinv() * camera.Vinv();
		// fill interleaved data
		vertexData[5 * nVertices] = wVertex.x;
		vertexData[5 * nVertices + 1] = wVertex.y;
		vertexData[5 * nVertices + 2] = 1; // red
		vertexData[5 * nVertices + 3] = 1; // green
		vertexData[5 * nVertices + 4] = 1; // blue
		nVertices++;
		// copy data to the GPU
		glBufferData(GL_ARRAY_BUFFER, nVertices * 5 * sizeof(float), vertexData, GL_DYNAMIC_DRAW);
	}

	void Draw() {
		if (nVertices > 0) {
			mat4 VPTransform = camera.V() * camera.P();

			int location = glGetUniformLocation(shaderProgram, "MVP");
			if (location >= 0) glUniformMatrix4fv(location, 1, GL_TRUE, VPTransform);
			else printf("uniform MVP cannot be set\n");

			glBindVertexArray(vao);
			glDrawArrays(GL_LINE_STRIP, 0, nVertices);
		}
	}
};

class LagrangeCurve : public LineStrip {
	std::vector<vec4> cps;
	std::vector<float> ts;
	std::vector<vec4> points;

	float firstClickTime;

	float L(int i, float t) {
		float Li = 1.0f;
		for (unsigned int j = 0; j < cps.size(); j++) {
			if (j != i) {
				Li *= ((t - ts[j]) / (ts[i] - ts[j]));
			}
		}
		return Li;
	}

public:

	std::vector<float> getTs() const { return ts; }

	std::vector<vec4> getCps() const { return cps; }

	std::vector<vec4> getPoints() const { return points; }

	LagrangeCurve() { 
		//printf("constructor ran\n");
		nVertices = 0; }

	void AddControlPoint(vec4 cp) {

		//printf("cp added\n");
		
		cps.push_back(cp);
		ts.push_back(ts.size());

		nVertices = 0;
		points.clear();

		for (unsigned int i = 0; i < cps.size() - 1; ++i) {

			float dt = (ts[i + 1] - ts[i]) / 100.0f;
			for (float t = ts[i]; t < ts[i + 1]; t += dt) {

				vec4 currentPoint = r(t);

				AddPoint(currentPoint.x, currentPoint.y);
				points.push_back(currentPoint);

			}
		}

	}

	vec4 r(float t) {
		vec4 rr(0, 0, 0, 1);
		for (unsigned int i = 0; i < cps.size(); i++)
			rr = rr + (cps[i] * L(i, t));
		return rr;
	}
	/*
	void startingPoints() {

		std::vector<vec4> pontok;
		const int csucsok_szama = 10;		// ötágú lesz a csillag
		float small_radius = 0.1;
		float big_radius = 0.3;
		
		pontok.push_back(vec4(0, small_radius));
		
		pontok.push_back(0);	// cX
		pontok.push_back(small_radius);  // cY
		
		for (int i = 0; i < csucsok_szama; i++)
		{
			float rad = (float)i / (float)csucsok_szama * 2.0f * M_PI;
			float radius = i % 2 == 0 ? small_radius : big_radius;
			float x = cosf(rad) * radius;
			float y = sinf(rad) * radius;

			//TESTING
			printf("radius: %f\n", radius);

			pontok.push_back(vec4(x, y));
			/*
			pontok.push_back(x);
			pontok.push_back(y);
			
		}

		//megvan a "pontok" vektor

		//körvonal elsõ pontja kell még egyszer...??
		//pontok.push_back(pontok.at(0));
		
		
		pontok.push_back(pontok.at(2));
		pontok.push_back(pontok.at(3));
		

		AddControlPoint(pontok.at(1));
		AddControlPoint(pontok.at(2));
		AddControlPoint(pontok.at(3));
		
		//AddControlPoint(vec4(pontok.at(0), pontok.at(1)));
		AddControlPoint(vec4(pontok.at(2), pontok.at(3)));
		AddControlPoint(vec4(pontok.at(4), pontok.at(5)));
		AddControlPoint(vec4(pontok.at(6), pontok.at(7)));
		
	} */
};

class TriangleFan {

protected:
	unsigned int vao;	// vertex array object id
	float phi;			// rotation
	float sx, sy;		// scaling
	float wTx, wTy;		// translation
	std::vector<float> vertexCoords;
	std::vector<float> vertexColors;
	float Red = 0.0f;
	float Green = 0.0f;
	float Blue = 1.0f;

	virtual void setVertexCoords() {
		
		const int num_cps = 100;
		//const int array_size = (num_cps + 2) * 2;

		//std::vector<float> vertexCoords;
		vertexCoords.push_back(0.0f);	// cX
		vertexCoords.push_back(0.0f);  // cY

		for (int i = 0; i < num_cps; i++)
		{
			float rad = (float)i / (float)num_cps * 2.0f * M_PI;
			float x = cosf(rad); // *3;
			float y = sinf(rad); // *3;

			vertexCoords.push_back(x);
			vertexCoords.push_back(y);
		}

		//körvonal elsõ pontja kell még egyszer...
		vertexCoords.push_back(vertexCoords.at(2));
		vertexCoords.push_back(vertexCoords.at(3));

		//return vertexCoords;
	}

	void setVertexColors() {
		//a vertexCoords összes eleméhez felveszünk 3 számot (RGB)
		for (int i = 0; i < vertexCoords.size(); i++) {
			vertexColors.push_back(Red);
			vertexColors.push_back(Green);
			vertexColors.push_back(Blue);
		}

	}
	
public:
	TriangleFan() {
		Animate(0);
	}

	void Create() {
		
		setVertexCoords();
			//std::vector<float> vertexCoords = getVertexCoords();
		
		setVertexColors();
		
		glGenVertexArrays(1, &vao);	// create 1 vertex array object
		glBindVertexArray(vao);		// make it active

		unsigned int vbo[2];		// vertex buffer objects
		glGenBuffers(2, &vbo[0]);	// Generate 2 vertex buffer objects

									// vertex coordinates: vbo[0] -> Attrib Array 0 -> vertexPosition of the vertex shader
		glBindBuffer(GL_ARRAY_BUFFER, vbo[0]); // make it active, it is an array
		//float vertexCoords[] = { 0,0,  -1,1,  1,1,  1,-1,  -1,-1,  -1,1 };
		//float vertexCoords[] = { -2, -2,   -1, 2,   2, -1,  5, -5,  6, 2 };	// vertex data on the CPU
		glBufferData(GL_ARRAY_BUFFER,      // copy to the GPU
			vertexCoords.size() * sizeof(float), //sizeof(vertexCoords), // number of the vbo in bytes
			&vertexCoords[0], //vertexCoords,		   // address of the data array on the CPU
			GL_STATIC_DRAW);	   // copy to that part of the memory which is not modified 
								   // Map Attribute Array 0 to the current bound vertex buffer (vbo[0])
		glEnableVertexAttribArray(0);
		// Data organization of Attribute Array 0 
		glVertexAttribPointer(0,			// Attribute Array 0
			2, GL_FLOAT,  // components/attribute, component type
			GL_FALSE,		// not in fixed point format, do not normalized
			0, NULL);     // stride and offset: it is tightly packed

						  // vertex colors: vbo[1] -> Attrib Array 1 -> vertexColor of the vertex shader
		glBindBuffer(GL_ARRAY_BUFFER, vbo[1]); // make it active, it is an array
		//float vertexColors[] = { 0, 0, 1,   0, 0, 1,   0, 0, 1 };	// vertex data on the CPU
		glBufferData(GL_ARRAY_BUFFER, vertexColors.size() * sizeof(float), &vertexColors[0], GL_STATIC_DRAW);	// copy to the GPU

																							// Map Attribute Array 1 to the current bound vertex buffer (vbo[1])
		glEnableVertexAttribArray(1);  // Vertex position
									   // Data organization of Attribute Array 1
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, NULL); // Attribute Array 1, components/attribute, component type, normalize?, tightly packed
	}

	void Animate(float t) {
		sx = 1; // sinf(t);
		sy = 1; // cosf(t);
		wTx = 0; // 4 * cosf(t / 2);
		wTy = 0; // 4 * sinf(t / 2);
		phi = 0; // t;
	}
	
	void Draw() {
		/*
		mat4 MVPTransform(0.1 * cos(phi), 0.1 * sin(phi), 0, 0,
						 -0.1 * sin(phi), 0.1 * cos(phi), 0, 0,
						  0, 0, 1, 0,
						  0, 0, 0, 1);
		*/

		mat4 Mscale(
			sx, 0, 0, 0,
			0, sy, 0, 0,
			0, 0, 0, 0,
			0, 0, 0, 1); // model matrix

		mat4 MRotate(
			 cos(phi), sin(phi), 0, 0,
			-sin(phi), cos(phi), 0, 0, 
			 0, 0, 1, 0,
			 0, 0, 0, 1);
		
		mat4 Mtranslate(
			1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 0, 0,
			wTx, wTy, 0, 1); // model matrix

		mat4 MVPTransform = Mscale * MRotate * Mtranslate * camera.V() * camera.P();
		// set GPU uniform matrix variable MVP with the content of CPU variable MVPTransform
		int location = glGetUniformLocation(shaderProgram, "MVP");
		if (location >= 0) glUniformMatrix4fv(location, 1, GL_TRUE, MVPTransform); // set uniform variable MVP to the MVPTransform
		else printf("uniform MVP cannot be set\n");

		glBindVertexArray(vao);	// make the vao and its vbos active playing the role of the data source
		glDrawArrays(GL_TRIANGLE_FAN, 0, vertexCoords.size() / 2);	// draw a single triangle with vertices defined in vao
	}
};

class Circle : public TriangleFan {
	// ami paraméterezhetõ kell legyen, az az eltolás, sugárméret

protected:
	float radius;

	void setVertexCoords() {
		const int num_cps = 100;
		//const int array_size = (num_cps + 2) * 2;

		//std::vector<float> vertexCoords;
		vertexCoords.push_back(0);	// cX
		vertexCoords.push_back(0);  // cY

		for (int i = 0; i < num_cps; i++)
		{
			float rad = (float)i / (float)num_cps * 2.0f * M_PI;
			float x = cosf(rad) * radius;
			float y = sinf(rad) * radius;

			vertexCoords.push_back(x);
			vertexCoords.push_back(y);
		}

		//körvonal elsõ pontja kell még egyszer...
		vertexCoords.push_back(vertexCoords.at(2));
		vertexCoords.push_back(vertexCoords.at(3));

	}

public:
	Circle(float x, float y, float r) {
		wTx = x;
		wTy = y;
		radius = r;
	}


};

class MyEllipse : public Circle {
	// sajátossága, hogy az X-scaling állítható

public:
	MyEllipse(float x, float y, float r, float sX) :Circle(x, y, r) {
		sx = sX;
		//Red = 1.0f;
	}
};

class Butterfly {
	//consists of body, head and wings
	// body: ellipse - scaling a circle - triangle_fan
	// head: circle - triangle_fan
	// wings: textured something...
};

class Petal {

	//ezek 1-1, 3 kontrollponttal megadott lagrange(?) görbét tartalmaznak - egyelõre
	LagrangeCurve lagrange;

public:
	void Create() {
		lagrange.Create();
	}

	void Draw() {
		lagrange.Draw();
	}

	void AddControlPoint(vec4 v) {
		lagrange.AddControlPoint(v);
	}

};

//Petal petal, petal2, petal3;
//TERV: manuális módszer, felveszünk (3+5+8+13) = 29 darab Petal-t........


class Flower {
	//consists of petals and "centre"
	// centre: circle - a triangle_fan
	// petals: ? - triangle_fan?
	// ötlet a kirajzoláshoz: elõször kirajzoljuk az egész csillagot, amit a szirmok alkotnak, majd FÖLÉ rajzoljuk a centert
	
	std::vector<Petal*> petals;
	static const int max_num_petals = 13; // maximum érték
	int num_petals;	//amennyit a maximumból "felhasználunk"
	std::vector<vec4> pontok;
	Petal petals_array[max_num_petals];

public:

	// paraméter: hány szirom legyen
	Flower(int n) {
		
		num_petals = n;

		
		//array //- csak hogy állandó memóriacímen legyenek, nem "dangling"
		for (int i = 0; i < n; i++)
		{
			Petal pe;
			petals_array[i] = pe;
		}
		

		
		//vector
		for (int i = 0; i < n; i++) {
			//Petal pe;						// trying to access a dangling pointer!!!!

			petals.push_back(&(petals_array[i]));
		}
		

		/*
		petals.push_back(&petal);
		petals.push_back(&petal2);
		petals.push_back(&petal3);
		*/

		/*
		//default "value"-k a petals tömbnek
		for (int i = 0; i < max_num_petals; i++) {
			Petal p1;
			petals[i] = p1;	//kell?
		}
		*/
	}

private:
	void buildPoints(int n) {

		const int csucsok_szama = 2 * num_petals; //10 - ötágú lesz a csillag
		float small_radius = 0.1;
		float big_radius = 0.3;

		for (int i = 0; i < csucsok_szama; i++)
		{
			float rad = (float)i / (float)csucsok_szama * 2.0f * M_PI;
			float radius = i % 2 == 0 ? big_radius : small_radius;
			float x = cosf(rad) * radius;
			float y = sinf(rad) * radius;

			pontok.push_back(vec4(x, y));
			//printf("Point added, i=%d, X: %f, Y: %f\n", i, x, y);
		}
	}

public:
	void buildPetals() {

		//létrehozunk n db megfelelõ paraméterekkel rendelkezõ szirmot

		buildPoints(2 * num_petals); // pontokat felvesszük - EGYELÕRE 6!!!!

		//ciklikusan felépítjük a szirmokat...
		// (1,2,3), (3,4,5), (5,6,7), ... , (9,0,1)
		// ha mondjuk 6 pont van -> 3 szirom, (1,2,3), (3,4,5), (5,0,1)

		
		for (int i = 0; i < num_petals; i++) {
		// 0 - (1,2,3)
		// 1 - (3,4,5)
		// 2 - (5,0,1)
		//int cps_to_add[] = { 2*i + 1, 2*i + 2, 2*i + 3};
		
			if (i == num_petals - 1) {
				// ilyenkor jön a (num_petals - 1, 0, 1) sorozat
				petals.at(i)->AddControlPoint(pontok.at(2 * i + 1));
				petals.at(i)->AddControlPoint(pontok.at(0));
				petals.at(i)->AddControlPoint(pontok.at(1));
			}
			else {
				petals.at(i)->AddControlPoint(pontok.at(2 * i + 1));
				petals.at(i)->AddControlPoint(pontok.at(2 * i + 2));
				petals.at(i)->AddControlPoint(pontok.at(2 * i + 3));
			}

		}
		

		//paraméteresen ^

		/*
		petals.at(0)->AddControlPoint(pontok.at(1));
		petals.at(0)->AddControlPoint(pontok.at(2));
		petals.at(0)->AddControlPoint(pontok.at(3));
		
		petals.at(1)->AddControlPoint(pontok.at(3));
		petals.at(1)->AddControlPoint(pontok.at(4));
		petals.at(1)->AddControlPoint(pontok.at(5));

		petals.at(2)->AddControlPoint(pontok.at(5));
		petals.at(2)->AddControlPoint(pontok.at(0));
		petals.at(2)->AddControlPoint(pontok.at(1));
		*/

		/*
		Petal p1, p2, p3;
		p1.AddControlPoint(pontok.at(1));
		p1.AddControlPoint(pontok.at(2));
		p1.AddControlPoint(pontok.at(3));
		
		p2.AddControlPoint(pontok.at(3));
		p2.AddControlPoint(pontok.at(4));
		p2.AddControlPoint(pontok.at(5));
		
		p3.AddControlPoint(pontok.at(5));
		p3.AddControlPoint(pontok.at(0));
		p3.AddControlPoint(pontok.at(1));
		
		petals.push_back(&p1);
		petals.push_back(&p2);
		petals.push_back(&p3);
		*/
		/*
		petal.AddControlPoint(pontok.at(1));
		petal.AddControlPoint(pontok.at(2));
		petal.AddControlPoint(pontok.at(3));

		petal2.AddControlPoint(pontok.at(3));
		petal2.AddControlPoint(pontok.at(4));
		petal2.AddControlPoint(pontok.at(5));
		
		
		petal3.AddControlPoint(pontok.at(5));
		petal3.AddControlPoint(pontok.at(0));
		petal3.AddControlPoint(pontok.at(1));
		*/
	}

	void Create() {

		//printf("Flower created, petals.size = %d\n", petals.size());
		// mindig 0 -> adjunk hozzá defaultokat, és utána módosítsuk azokat! (addControlPoint)

		//TESZT
		//buildPetals();

		/*
		for (int i = 0; i < num_petals; i++)
		{
			petals_array[i]->Create();
		}
		*/

		
		//szirmok
		for (int i = 0; i < petals.size(); i++) {
			Petal *pointer = petals.at(i);
			//Petal p = petals.at(i);
			//p.Create();
			pointer->Create();
		}
		

		//center
		// TODO
	}

	void Draw() {
		/*
		for (int i = 0; i < num_petals; i++)
		{
			petals_array[i]->Draw();
		}
		*/
		
		for (int i = 0; i < petals.size(); i++) {
			Petal *p = petals.at(i);
			p->Draw();
			//Petal p = petals.at(i);
			//p.Draw();
		}
		
	}

};


// The virtual world
TriangleFan triangle;
Circle fc(0, 0, 0.5);
Circle fc2(4, 5, 0.5);
MyEllipse ell(0, 0, 2, 0.2);
Circle head(0, 2.5, 0.5);
//LagrangeCurve lagrange;
Flower flower(13);
Flower flower2(8);
Flower flower3(5);
Flower flower4(3);


// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);

	// Create objects by setting up their vertex data on the GPU
	triangle.Create();
	fc.Create();
	fc2.Create();
	ell.Create();
	head.Create();

	/*
	//TESZT
	petal.Create();
	petal2.Create();
	petal3.Create();
	*/

	flower.Create();
	flower.buildPetals();

	flower2.Create();
	flower2.buildPetals();

	flower3.Create();
	flower3.buildPetals();

	flower4.Create();
	flower4.buildPetals();

	//lagrange.Create();

	//lagrange.startingPoints();
	
	/*
	lagrange.AddControlPoint(vec4(-0.05, -0.1));
	lagrange.AddControlPoint(vec4(0, 0));
	lagrange.AddControlPoint(vec4(0.05, -0.1));
	*/

	// Create vertex shader from string
	unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
	if (!vertexShader) { printf("Error in vertex shader creation\n"); exit(1); }
	glShaderSource(vertexShader, 1, &vertexSource, NULL);
	glCompileShader(vertexShader);
	checkShader(vertexShader, (char*)"Vertex shader error");

	// Create fragment shader from string
	unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
	if (!fragmentShader) { printf("Error in fragment shader creation\n"); exit(1); }
	glShaderSource(fragmentShader, 1, &fragmentSource, NULL);
	glCompileShader(fragmentShader);
	checkShader(fragmentShader, (char*)"Fragment shader error");

	// Attach shaders to a single program
	shaderProgram = glCreateProgram();
	if (!shaderProgram) { printf("Error in shader program creation\n"); exit(1); }
	glAttachShader(shaderProgram, vertexShader);
	glAttachShader(shaderProgram, fragmentShader);

	// Connect the fragmentColor to the frame buffer memory
	glBindFragDataLocation(shaderProgram, 0, "fragmentColor");	// fragmentColor goes to the frame buffer memory

																// program packaging
	glLinkProgram(shaderProgram);
	checkLinking(shaderProgram);
	// make this program run
	glUseProgram(shaderProgram);
}

void onExit() {
	glDeleteProgram(shaderProgram);
	printf("exit");
}

// Window has become invalid: Redraw
void onDisplay() {
	glClearColor(0, 1, 0, 0);							// background color 
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear the screen
	//triangle.Draw();
	//fc.Draw();
	//fc2.Draw();
	//ell.Draw();
	//head.Draw();

	flower.Draw();
	flower2.Draw();
	flower3.Draw();
	flower4.Draw();

	/*
	petal.Draw(); 
	petal2.Draw(); 
	petal3.Draw();
	*/

	//lagrange.Draw();
	
	glutSwapBuffers();									// exchange the two buffers
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == 'd') glutPostRedisplay();         // if d, invalidate display, i.e. redraw
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {
}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) {
	if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {  // GLUT_LEFT_BUTTON / GLUT_RIGHT_BUTTON and GLUT_DOWN / GLUT_UP
		float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
		float cY = 1.0f - 2.0f * pY / windowHeight;
		//lagrange.AddControlPoint(vec4(cX, cY));
		glutPostRedisplay();     // redraw
	}
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	long time = glutGet(GLUT_ELAPSED_TIME); // elapsed time since the start of the program
	float sec = time / 1000.0f;				// convert msec to sec
	triangle.Animate(sec);					// animate the triangle object
	glutPostRedisplay();					// redraw the scene
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Do not touch the code below this line

int main(int argc, char * argv[]) {
	glutInit(&argc, argv);
#if !defined(__APPLE__)
	glutInitContextVersion(majorVersion, minorVersion);
#endif
	glutInitWindowSize(windowWidth, windowHeight);				// Application window is initially of resolution 600x600
	glutInitWindowPosition(100, 100);							// Relative location of the application window
#if defined(__APPLE__)
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH | GLUT_3_3_CORE_PROFILE);  // 8 bit R,G,B,A + double buffer + depth buffer
#else
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
#endif
	glutCreateWindow(argv[0]);

#if !defined(__APPLE__)
	glewExperimental = true;	// magic
	glewInit();
#endif

	printf("GL Vendor    : %s\n", glGetString(GL_VENDOR));
	printf("GL Renderer  : %s\n", glGetString(GL_RENDERER));
	printf("GL Version (string)  : %s\n", glGetString(GL_VERSION));
	glGetIntegerv(GL_MAJOR_VERSION, &majorVersion);
	glGetIntegerv(GL_MINOR_VERSION, &minorVersion);
	printf("GL Version (integer) : %d.%d\n", majorVersion, minorVersion);
	printf("GLSL Version : %s\n", glGetString(GL_SHADING_LANGUAGE_VERSION));

	onInitialization();

	glutDisplayFunc(onDisplay);                // Register event handlers
	glutMouseFunc(onMouse);
	glutIdleFunc(onIdle);
	glutKeyboardFunc(onKeyboard);
	glutKeyboardUpFunc(onKeyboardUp);
	glutMotionFunc(onMouseMotion);

	glutMainLoop();
	onExit();
	return 1;
}
