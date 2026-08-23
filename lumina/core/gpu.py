"""OpenGL render engine — the develop pipeline as GLSL shaders.

Interactive edits become uniform updates + a redraw (<10 ms typical),
with the CPU numpy pipeline retained for exports and fallback.
"""
from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import QRectF
from PySide6.QtGui import QImage
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import QApplication

VS_QUAD = """
attribute vec2 aPos;
varying vec2 vUV;
uniform vec4 uRect;      /* pixels in widget coords */
uniform vec2 uViewport;  /* widget w,h */
void main() {
    vUV = aPos * 0.5 + 0.5;
    float px = uRect.x + (aPos.x * 0.5 + 0.5) * uRect.z;
    float py = uRect.y + (aPos.y * 0.5 + 0.5) * uRect.w;
    gl_Position = vec4(px / uViewport.x * 2.0 - 1.0,
                       1.0 - py / uViewport.y * 2.0, 0.0, 1.0);
}
"""

FS_FULL = """  /* offscreen full-frame passes */
varying vec2 vUV;
uniform sampler2D uBase;
uniform sampler2D uCurve;
uniform float uTemp, uTint, uExposure, uContrast;
uniform float uHighlights, uShadows, uWhites, uBlacks;
uniform float uVibrance, uSaturation, uBW;
uniform vec3 uBWW;
uniform float uH[8];
uniform float uS[8];
uniform float uL[8];
uniform vec3 uGSh, uGMt, uGHi;
uniform float uBlender, uBalance;

float lum(vec3 c){ return dot(c, vec3(0.2126, 0.7152, 0.0722)); }
float sstep(float a, float b, float x){
    float t = clamp((x - a) / max(b - a, 1e-5), 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

vec3 rgb2hsv(vec3 c){
    vec4 K = vec4(0.0, -1.0/3.0, 2.0/3.0, -1.0);
    vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
    vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));
    float d = q.x - min(q.w, q.y);
    return vec3(abs(q.z + (q.w - q.y) / (6.0*d + 1e-10)), d / (q.x + 1e-10), q.x);
}
vec3 hsv2rgb(vec3 c){
    vec4 K = vec4(1.0, 2.0/3.0, 1.0/3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

float bandW(float hue, float lo, float span, float fall){
    float d = mod(hue - lo, 360.0);
    float inside = sstep(-fall*0.5, fall*0.5 + 0.001, d)
                 * (1.0 - sstep(span - fall*0.5, span + fall*0.5 + 0.001, d));
    float core = sstep(-fall*0.15, fall*0.15, d)
               * (1.0 - sstep(span - fall*0.15, span + fall*0.15, d));
    return max(inside * 0.55, core);
}

void main() {
    vec3 col = texture2D(uBase, vUV).rgb;

    /* white balance */
    col *= vec3(1.0 + 0.28*uTemp, 1.0 - 0.14*uTint, 1.0 - 0.28*uTemp);
    col = clamp(col, 0.0, 1.0);

    /* exposure in linear light */
    col = pow(col, vec3(2.2)) * exp2(uExposure);
    col = clamp(pow(clamp(col, 0.0, 1.0), vec3(1.0/2.2)), 0.0, 1.0);

    /* contrast — sigmoid blend like CPU path */
    float l0 = lum(col);
    if (uContrast >= 0.0) {
        float k = 1.0 + 6.0*uContrast;
        float sig = 1.0 / (1.0 + exp(-(l0 - 0.5) * k));
        col = mix(col, vec3(sig), uContrast);
    } else {
        col = 0.5 + (col - 0.5) * (1.0 + uContrast * 0.85);
    }

    /* tone region ops */
    float l = lum(col);
    vec3 wH = vec3(pow(sstep(0.35, 0.95, l), 1.2));
    if (uHighlights < 0.0) col += uHighlights * 0.75 * wH * (col - l*0.6);
    else                   col += uHighlights * 0.55 * wH * (vec3(1.0) - col);

    vec3 wS = vec3(pow(1.0 - sstep(0.0, 0.55, l), 1.4));
    if (uShadows > 0.0) col += uShadows * 0.62 * wS * (vec3(1.0) - col);
    else                col += uShadows * 0.55 * wS * col;

    vec3 wWt = vec3(pow(sstep(0.45, 1.0, l), 2.0));
    if (uWhites > 0.0) col += uWhites * 0.45 * wWt * (vec3(1.0) - col);
    else               col += uWhites * 0.40 * wWt * col;

    vec3 wB = vec3(pow(1.0 - sstep(0.02, 0.42, l), 2.0));
    if (uBlacks < 0.0) col += uBlacks * 0.40 * wB * col;
    else col += uBlacks * 0.42 * wB * clamp(vec3(0.35) - col, vec3(-0.35), vec3(0.35)) * 1.4;

    /* curves: R=composite, G=R-curve, B=G-curve, A=B-curve */
    for (int i = 0; i < 3; ++i) {
        col[i] = texture2D(uCurve, vec2(clamp(col[i], 0.0, 1.0), 0.5)).r;
        float cr = texture2D(uCurve, vec2(clamp(col[i], 0.0, 1.0), 0.5))[i+1];
        col[i] = cr;
    }
    col = clamp(col, 0.0, 1.0);

    /* vibrance & saturation */
    vec3 hsv = rgb2hsv(col);
    hsv.y = clamp(hsv.y * (1.0 + uVibrance * (1.0 - hsv.y) * 1.1), 0.0, 1.0);
    col = hsv2rgb(hsv);
    l = lum(col);
    col = clamp(l + (col - l) * (1.0 + uSaturation), 0.0, 1.0);

    /* HSL mixer */
    hsv = rgb2hsv(col);
    if (hsv.y > 0.004 && hsv.z > 0.004) {
        float hueShift = 0.0; float satMul = 1.0; float lumAdd = 0.0;
        for (int b = 0; b < 8; ++b) {
            float dh = uH[b], ds = uS[b], dl = uL[b];
            if (dh == 0.0 && ds == 0.0 && dl == 0.0) continue;
            float lo, span, fall;
            if (b == 0) { lo=345.0; span=30.0; fall=22.0; }
            else if (b==1){ lo=12.0;  span=34.0; fall=18.0; }
            else if (b==2){ lo=38.0;  span=34.0; fall=18.0; }
            else if (b==3){ lo=58.0;  span=107.0;fall=30.0; }
            else if (b==4){ lo=150.0; span=55.0; fall=22.0; }
            else if (b==5){ lo=192.0; span=73.0; fall=24.0; }
            else if (b==6){ lo=252.0; span=53.0; fall=22.0; }
            else         { lo=292.0; span=58.0; fall=22.0; }
            float w = bandW(degrees(hsv.x*6.28318530718), lo, span, fall) * step(0.004, hsv.y);
            hueShift += dh * 0.32 * w;
            satMul   *= 1.0 + ds * 1.15 * w;
            lumAdd   += dl * 0.16 * w;
        }
        hsv.x = fract(hsv.x + hueShift / 360.0);
        hsv.y = clamp(hsv.y * satMul, 0.0, 1.0);
        col = hsv2rgb(hsv) + lumAdd;
    }

    /* black & white */
    if (uBW > 0.5) {
        float g = dot(col, uBWW);
        col = vec3(g);
    } else {
        col = clamp(col, 0.0, 1.0);
    }

    /* color grading wheels */
    float bal = uBalance;
    float center = 0.5 + bal * 0.25;
    float spread = 0.28 + uBlender * 0.34;
    float ws = pow(1.0 - sstep(center - spread*0.9, center + spread, l), 1.3);
    float wh = pow(sstep(center - spread, center + spread*0.9, l), 1.3);
    float sigma = spread * 0.62;
    float wm = pow(exp(-((l - center)*(l-center)) / (2.0*sigma*sigma)), 1.2);

    vec3 tsh = hsv2rgb(vec3(fract(uGSh.x/360.0), min(1.0, uGSh.y*1.6), 1.0));
    vec3 tmt = hsv2rgb(vec3(fract(uGMt.x/360.0), min(1.0, uGMt.y*1.6), 1.0));
    vec3 thi = hsv2rgb(vec3(fract(uGHi.x/360.0), min(1.0, uGHi.y*1.6), 1.0));

    float sth = uGSh.y * 0.42;
    if (sth > 0.005 || abs(uGSh.z) > 0.005) {
        vec3 tinted = col*(1.0-sth*0.85) + tsh*sth*0.85*col*1.7;
        tinted = clamp(tinted, 0.0, 1.0);
        col = mix(col, tinted, sth);
        col += uGSh.z * 0.28 * ws;
    }
    float smt = uGMt.y * 0.42;
    if (smt > 0.005 || abs(uGMt.z) > 0.005) {
        float bell = col.r*(1.0-col.r)*4.0 + col.g*(1.0-col.g)*4.0 + col.b*(1.0-col.b)*4.0;
        bell /= 3.0;
        vec3 tinted = clamp(col + (tmt - 0.5) * smt * bell, 0.0, 1.0);
        col = mix(col, tinted, smt);
        col += uGMt.z * 0.28 * wm;
    }
    float shi = uGHi.y * 0.42;
    if (shi > 0.005 || abs(uGHi.z) > 0.005) {
        vec3 tinted = clamp(col + (thi - 0.5) * shi * col * 1.6, 0.0, 1.0);
        col = mix(col, tinted, shi);
        col += uGHi.z * 0.28 * wh;
    }

    gl_FragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
"""

FS_BLUR = """
varying vec2 vUV;
uniform sampler2D uTex;
uniform vec2 uDir;       /* texel step direction */
void main() {
    vec4 sum = texture2D(uTex, vUV) * 0.227027;
    sum += texture2D(uTex, vUV + uDir * 1.3846) * 0.316216;
    sum += texture2D(uTex, vUV - uDir * 1.3846) * 0.316216;
    sum += texture2D(uTex, vUV + uDir * 3.2308) * 0.070270;
    sum += texture2D(uTex, vUV - uDir * 3.2308) * 0.070270;
    gl_FragColor = sum;
}
"""

FS_DETAIL = """
varying vec2 vUV;
uniform sampler2D uA;      /* base result */
uniform sampler2D uBs;     /* small blur of A */
uniform sampler2D uBb;     /* big blur of A */
uniform sampler2D uNoise;
uniform float uSharpAmt, uSharpRad, uNrLum, uNrColor;
uniform float uClar;
uniform float uVigAmt, uVigMid, uVigFeather;
uniform float uGrainAmt;
uniform vec2 uSeedOff;

float lum(vec3 c){ return dot(c, vec3(0.2126, 0.7152, 0.0722)); }
float sstep(float a, float b, float x){
    float t = clamp((x - a) / max(b - a, 1e-5), 0.0, 1.0);
    return t*t*(3.0 - 2.0*t);
}

void main() {
    vec3 col = texture2D(uA, vUV).rgb;
    vec3 bs = texture2D(uBs, vUV).rgb;
    vec3 bb = texture2D(uBb, vUV).rgb;
    float l = lum(col);

    /* chroma noise reduction */
    float lbs = lum(bs);
    vec3 chromaB = bs - lbs;
    col = lbs + (col - l) + (chromaB - (col - l)) * uNrColor;
    l = lum(col);

    /* luminance NR (edge aware) */
    float keep = sstep(0.015, 0.09, abs(l - lbs));
    float lnew = mix(lbs, l, keep);
    col += (lnew - l);
    l = lnew;

    /* clarity (midtone local contrast via big blur) */
    float midw = pow(1.0 - abs(2.0*sstep(0.0, 1.0, l) - 1.0), 0.7);
    float detailBig = l - lum(bb);
    col += (uClar * 0.9 * midw * detailBig);

    /* sharpening (small blur unsharp on luma) */
    col += uSharpAmt * 1.35 * (l - lbs);

    /* vignette */
    float vig = 0.0;
    if (abs(uVigAmt) > 0.001) {
        float m = max(0.02, uVigMid);
        float f = max(0.02, uVigFeather);
        float end = m + (1.0 - m) * f;
        vec2 pc = (vUV - 0.5) * vec2(1.0, 1.0);
        float d = length(pc) / 0.7071;
        float fall = sstep(m, end, d); fall = fall*fall;
        vig = uVigAmt * 1.15 * fall;
        col *= (1.0 + vig);
    }

    /* grain */
    if (uGrainAmt > 0.001) {
        float n = texture2D(uNoise, vUV * 0.31 + uSeedOff).r - 0.5;
        col += n * uGrainAmt * 0.16;
    }

    gl_FragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
"""

FS_PRESENT = """   /* final screen draw through crop window */
varying vec2 vUV;
uniform sampler2D uIn;
uniform vec4 uCrop;     /* uv window into rendered frame */
void main() {
    vec2 uv = vec2(mix(uCrop.x, uCrop.z, vUV.x),
                   mix(uCrop.w, uCrop.y, vUV.y));
    gl_FragColor = vec4(texture2D(uIn, uv).rgb, 1.0);
}
"""



import ctypes as _ctypes


def _gen1(glf, kind: str) -> int:
    """PySide6 QOpenGLFunctions requires the C-style (count, out_array) form."""
    arr = (_ctypes.c_uint * 1)()
    if kind == "tex":
        glf.glGenTextures(1, arr)
    elif kind == "buf":
        glf.glGenBuffers(1, arr)
    elif kind == "fbo":
        glf.glGenFramebuffers(1, arr)
    else:
        raise ValueError(kind)
    return int(arr[0])


class _FBO:
    """Render target: QOpenGLTexture storage attached to a raw FBO.
    (Raw glTexImage2D with Python buffers deadlocks the Metal-GL bridge;
    QOpenGLTexture allocation is safe and fast.)"""

    def __init__(self):
        self.qtex = None
        self.fbo = None
        self.w = 0
        self.h = 0

    @property
    def tex(self):
        return self.qtex.textureId() if self.qtex is not None else None

    def ensure(self, glf, w, h):
        if self.fbo is not None and self.w == w and self.h == h:
            return
        self.release(glf)
        self.w, self.h = w, h
        from PySide6.QtOpenGL import QOpenGLTexture
        t = QOpenGLTexture(QOpenGLTexture.Target2D)
        t.setSize(w, h)
        t.setFormat(QOpenGLTexture.RGB8_UNorm)
        t.setMinificationFilter(QOpenGLTexture.Linear)
        t.setMagnificationFilter(QOpenGLTexture.Linear)
        t.setWrapMode(QOpenGLTexture.ClampToEdge)
        t.allocateStorage()
        self.qtex = t
        self.fbo = _gen1(glf, "fbo")
        glf.glBindFramebuffer(0x8D40, self.fbo)
        glf.glFramebufferTexture2D(0x8D40, 0x8CE0, 0x0DE1, t.textureId(), 0)
        glf.glBindFramebuffer(0x8D40, 0)

    def release(self, glf):
        from PySide6.QtOpenGL import QOpenGLTexture
        if isinstance(self.qtex, QOpenGLTexture):
            self.qtex.destroy()
        self.qtex = None
        if self.fbo:
            glf.glDeleteFramebuffers(int(self.fbo))
        self.fbo = None


class GLRenderer:
    """Owns programs/FBOs/textures; must be used with a current GL context."""

    def __init__(self):
        self.ok = False
        self.error = ""
        self._progs = {}
        self.base_tex = None
        self.curve_tex = None
        self.noise_tex = None
        self.mask_texes = {}          # mask_id -> (tex, revision_key)
        self.A, self.T, self.Bs, self.Bs2, self.Bb, self.Bb2, self.OUT = (
            _FBO(), _FBO(), _FBO(), _FBO(), _FBO(), _FBO(), _FBO())
        self.w = self.h = 0
        self._quad_vbo = None
        self._glf = None

    # ------------------------------------------------------------ setup
    def setup(self, ctx) -> bool:
        from PySide6.QtOpenGL import QOpenGLShaderProgram, QOpenGLShader
        glf = ctx.functions()
        self._glf = glf
        try:
            for name, vs, fs in (
                ("full", VS_QUAD, FS_FULL),
                ("blur", VS_QUAD, FS_BLUR),
                ("detail", VS_QUAD, FS_DETAIL),
                ("present", VS_QUAD, FS_PRESENT),
            ):
                prog = QOpenGLShaderProgram()
                prog.addShaderFromSourceCode(QOpenGLShader.Vertex, vs)
                prog.addShaderFromSourceCode(QOpenGLShader.Fragment, fs)
                prog.bindAttributeLocation("aPos", 0)
                prog.link()
                self._progs[name] = prog
            glf.glClearColor(0.098, 0.098, 0.098, 1.0)
            glf.glDisable(0x0BE2)  # GL_BLEND
            # quad vertices as client-side array (no VBO needed on GL 2.1)
            from shiboken6 import VoidPtr
            import ctypes as _ct
            self._quad_arr = (_ct.c_float * 8)(-1.0, -1.0, 1.0, -1.0,
                                               -1.0, 1.0, 1.0, 1.0)
            self._quad_ptr = VoidPtr(_ct.addressof(self._quad_arr))
            # noise texture
            rng = np.random.default_rng(12345)
            noise = (rng.random((512, 512, 4)) * 255).astype(np.uint8)
            img = QImage(noise.data, 512, 512, 512 * 4,
                         QImage.Format_RGBA8888).copy()
            self.noise_tex = self._tex_from_image(glf, img, linear=False)
            self.ok = True
        except Exception as e:
            self.error = str(e)
            self.ok = False
        return self.ok

    def _tex_from_image(self, glf, img: QImage, linear=True):
        """QOpenGLTexture avoids a PySide6 marshalling hang on >4MB raw uploads."""
        from PySide6.QtOpenGL import QOpenGLTexture
        tex = QOpenGLTexture(img)
        filt = QOpenGLTexture.Linear if linear else QOpenGLTexture.Nearest
        tex.setMinificationFilter(filt)
        tex.setMagnificationFilter(filt)
        tex.setWrapMode(QOpenGLTexture.ClampToEdge)
        return tex

    def upload_base(self, u8: np.ndarray):
        glf = self._glf
        h, w = u8.shape[:2]
        self.w, self.h = w, h
        from PySide6.QtOpenGL import QOpenGLTexture
        if isinstance(self.base_tex, QOpenGLTexture):
            self.base_tex.destroy()
        img = QImage(u8.data, w, h, w * 3, QImage.Format_RGB888).copy()
        self.base_tex = self._tex_from_image(glf, img)

    def upload_curve(self, rgba_u8: np.ndarray):
        glf = self._glf
        n = rgba_u8.shape[0]
        from PySide6.QtOpenGL import QOpenGLTexture
        if isinstance(self.curve_tex, QOpenGLTexture):
            self.curve_tex.destroy()
        img = QImage(rgba_u8.data, n, 1, n * 4, QImage.Format_RGBA8888).copy()
        self.curve_tex = self._tex_from_image(glf, img)

    def set_masks(self, masks: dict):
        """masks: {mask_id: float32 array HxW} — replaces all."""
        glf = self._glf
        from PySide6.QtOpenGL import QOpenGLTexture
        for old in list(self.mask_texes.values()):
            if isinstance(old, QOpenGLTexture):
                old.destroy()
            else:
                glf.glDeleteTextures(int(old))
        self.mask_texes.clear()
        for mid, arr in masks.items():
            h, w = arr.shape[:2]
            u8 = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
            img = QImage(u8.data, w, h, w, QImage.Format_Grayscale8).copy()
            self.mask_texes[mid] = self._tex_from_image(glf, img)

    # ------------------------------------------------------------ drawing
    def _bind_raw(self, tex_id, unit: int):
        """Bind a raw FBO texture id at unit without PySide6 marshalling."""
        glf = self._glf
        glf.glActiveTexture(0x84C0 + unit)
        glf.glBindTexture(0x0DE1, tex_id)

    def _bind_quad(self, prog, rect: QRectF, vw, vh):
        glf = self._glf
        glf.glVertexAttribPointer(0, 2, 0x1406, False, 0, self._quad_ptr)
        glf.glEnableVertexAttribArray(0)
        prog.setUniformValue("uRect", float(rect.x()), float(rect.y()),
                             float(rect.width()), float(rect.height()))
        prog.setUniformValue("uViewport", float(vw), float(vh))

    def _draw_quad(self, prog, rect: QRectF, vw, vh):
        glf = self._glf
        self._bind_quad(prog, rect, vw, vh)
        glf.glDrawArrays(0x0004, 0, 4)   # TRIANGLE_STRIP
        glf.glDisableVertexAttribArray(0)

    def _render_full_pass(self, s: dict, dest: _FBO):
        glf = self._glf
        prog = self._progs["full"]
        dest.ensure(glf, self.w, self.h)
        glf.glBindFramebuffer(0x8D40, dest.fbo)
        glf.glViewport(0, 0, self.w, self.h)
        prog.bind()
        self.base_tex.bind(0)
        prog.setUniformValue("uBase", 0)
        (self.curve_tex if self.curve_tex is not None else self.base_tex).bind(1)
        prog.setUniformValue("uCurve", 1)

        prog.setUniformValue("uTemp", float(s["temp"]) / 100.0)
        prog.setUniformValue("uTint", float(s["tint"]) / 100.0)
        prog.setUniformValue("uExposure", float(s["exposure"]))
        prog.setUniformValue("uContrast", float(s["contrast"]) / 100.0)
        prog.setUniformValue("uHighlights", float(s["highlights"]) / 100.0)
        prog.setUniformValue("uShadows", float(s["shadows"]) / 100.0)
        prog.setUniformValue("uWhites", float(s["whites"]) / 100.0)
        prog.setUniformValue("uBlacks", float(s["blacks"]) / 100.0)
        prog.setUniformValue("uVibrance", float(s["vibrance"]) / 100.0)
        prog.setUniformValue("uSaturation", float(s["saturation"]) / 100.0)
        bw = 1.0 if s.get("bw") else 0.0
        prog.setUniformValue("uBW", bw)
        hsl = s.get("bw") and s.get("hsl")
        wr, wg, wb_ = 0.2126, 0.7152, 0.0722
        if hsl:
            wr *= 1 + hsl["red"][2]/130.0
            wg *= 1 + hsl["green"][2]/130.0 + hsl["yellow"][2]/220.0
            wb_ *= 1 + hsl["blue"][2]/110.0 + hsl["aqua"][2]/190.0
        tot = wr + wg + wb_
        prog.setUniformValue("uBWW", wr/tot, wg/tot, wb_/tot)

        bands = ["red","orange","yellow","green","aqua","blue","purple","magenta"]
        hs, ss, ls_ = [], [], []
        for b in bands:
            v = s["hsl"][b]
            hs.append(float(v[0]) / 100.0)
            ss.append(float(v[1]) / 100.0)
            ls_.append(float(v[2]) / 100.0)
        try:
            prog.setUniformValueArray("uH", hs)
            prog.setUniformValueArray("uS", ss)
            prog.setUniformValueArray("uL", ls_)
        except Exception:
            for i in range(8):
                prog.setUniformValue(f"uH[{i}]", hs[i])
                prog.setUniformValue(f"uS[{i}]", ss[i])
                prog.setUniformValue(f"uL[{i}]", ls_[i])

        gsh = s["grade_shadows"]; gmt = s["grade_midtones"]; ghi = s["grade_highlights"]
        prog.setUniformValue("uGSh", float(gsh[0]), float(gsh[1])/100.0, float(gsh[2])/100.0)
        prog.setUniformValue("uGMt", float(gmt[0]), float(gmt[1])/100.0, float(gmt[2])/100.0)
        prog.setUniformValue("uGHi", float(ghi[0]), float(ghi[1])/100.0, float(ghi[2])/100.0)
        prog.setUniformValue("uBlender", float(s["grade_blender"]) / 100.0)
        prog.setUniformValue("uBalance", float(s["grade_balance"]) / 100.0)

        self._draw_quad(prog, QRectF(0, 0, self.w, self.h), self.w, self.h)
        prog.release()

    def render_frame_offscreen(self, s: dict, masks: list, seed_offset=(0.0, 0.0)):
        """Full chain into OUT fbo. masks: [(id, invert)] with textures set."""
        glf = self._glf
        # 1) global
        self._render_full_pass(s, self.A)

        # 2) per-mask variants blended
        prog_blend = self._make_blend_prog_once()
        for mid, invert in masks:
            tex = self.mask_texes.get(mid)
            if tex is None:
                continue
            # render variant into T using same settings but mask-adjustments merged
            ms = self._merge_mask_adj(s, masks, mid)
            self._render_full_pass(ms, self.T)
            # blend A,T -> A2 then swap
            prog = prog_blend
            tmp = self._scratch_fbo()
            tmp.ensure(glf, self.w, self.h)
            glf.glBindFramebuffer(0x8D40, tmp.fbo)
            glf.glViewport(0, 0, self.w, self.h)
            prog.bind()
            self._bind_raw(self.A.tex, 0); prog.setUniformValue("uA", 0)
            self._bind_raw(self.T.tex, 1); prog.setUniformValue("uB", 1)
            tex.bind(2); prog.setUniformValue("uMask", 2)
            prog.setUniformValue("uInvert", 1 if invert else 0)
            self._draw_quad(prog, QRectF(0, 0, self.w, self.h), self.w, self.h)
            prog.release()
            self.A, tmp = tmp, self.A
            self._scratch_pool.append(tmp)

        # 3) blurs
        self._blur_chain(self.A, self.Bs, self.Bs2, sigma_px=1.6)
        self._blur_chain(self.A, self.Bb, self.Bb2, sigma_px=max(6.0, self.w / 90.0))

        # 4) detail/final
        prog = self._progs["detail"]
        self.OUT.ensure(glf, self.w, self.h)
        glf.glBindFramebuffer(0x8D40, self.OUT.fbo)
        glf.glViewport(0, 0, self.w, self.h)
        prog.bind()
        self._bind_raw(self.A.tex, 0); prog.setUniformValue("uA", 0)
        self._bind_raw(self.Bs2.tex, 1); prog.setUniformValue("uBs", 1)
        self._bind_raw(self.Bb2.tex, 2); prog.setUniformValue("uBb", 2)
        self.noise_tex.bind(3)
        prog.setUniformValue("uNoise", 3)
        prog.setUniformValue("uSharpAmt", float(s["sharp_amount"]) / 100.0)
        rad_scale = max(0.5, min(3.0, float(s["sharp_radius"]) * (self.w / 1600.0)))
        prog.setUniformValue("uSharpRad", rad_scale)
        prog.setUniformValue("uNrLum", float(s["nr_lum"]) / 100.0)
        prog.setUniformValue("uNrColor", float(s["nr_color"]) / 100.0)
        prog.setUniformValue("uClar", float(s["clarity"]) / 100.0)
        prog.setUniformValue("uVigAmt", float(s["vignette_amount"]) / 100.0)
        prog.setUniformValue("uVigMid", float(s["vignette_midpoint"]) / 100.0)
        prog.setUniformValue("uVigFeather", float(s["vignette_feather"]) / 100.0)
        prog.setUniformValue("uGrainAmt", float(s["grain_amount"]) / 100.0)
        prog.setUniformValue("uSeedOff", float(seed_offset[0]), float(seed_offset[1]))
        self._draw_quad(prog, QRectF(0, 0, self.w, self.h), self.w, self.h)
        prog.release()
        glf.glActiveTexture(0x84C0)

    def draw_to_screen(self, crop_uv, display_rect: QRectF, view_w, view_h):
        glf = self._glf
        glf.glBindFramebuffer(0x8D40, 0)
        glf.glViewport(0, 0, int(view_w), int(view_h))
        glf.glClear(0x4100)   # COLOR|DEPTH
        prog = self._progs["present"]
        prog.bind()
        self._bind_raw(self.OUT.tex, 0)
        prog.setUniformValue("uIn", 0)
        from PySide6.QtGui import QVector4D
        prog.setUniformValue("uCrop", QVector4D(float(crop_uv[0]), float(crop_uv[1]),
                                                float(crop_uv[2]), float(crop_uv[3])))
        self._draw_quad(prog, display_rect, view_w, view_h)
        prog.release()

    # ------------------------------------------------------------ helpers
    _blend_prog = None
    _scratch_pool = []

    def _make_blend_prog_once(self):
        if GLRenderer._blend_prog is not None:
            return GLRenderer._blend_prog
        from PySide6.QtOpenGL import QOpenGLShaderProgram, QOpenGLShader
        prog = QOpenGLShaderProgram()
        prog.addShaderFromSourceCode(QOpenGLShader.Vertex, VS_QUAD)
        prog.addShaderFromSourceCode(QOpenGLShader.Fragment, """
varying vec2 vUV;
uniform sampler2D uA; uniform sampler2D uB; uniform sampler2D uMask;
uniform int uInvert;
void main(){
    float w = texture2D(uMask, vUV).r;
    if (uInvert == 1) w = 1.0 - w;
    vec3 a = texture2D(uA, vUV).rgb;
    vec3 b = texture2D(uB, vUV).rgb;
    gl_FragColor = vec4(mix(a, b, w), 1.0);
}
""")
        prog.bindAttributeLocation("aPos", 0)
        prog.link()
        GLRenderer._blend_prog = prog
        return prog

    def _scratch_fbo(self):
        if GLRenderer._scratch_pool:
            return GLRenderer._scratch_pool.pop()
        return _FBO()

    @staticmethod
    def _merge_mask_adj(s: dict, masks: list, mid) -> dict:
        out = dict(s)
        md = next((m for m in s.get("masks", []) if m.get("id") == mid), None)
        if md:
            for k, v in md.get("adjustments", {}).items():
                if k in out and isinstance(out[k], (int, float)):
                    out[k] = v
        return out

    def _blur_chain(self, src: _FBO, tmp: _FBO, dst: _FBO, sigma_px: float):
        glf = self._glf
        prog = self._progs["blur"]
        tmp.ensure(glf, self.w, self.h)
        dst.ensure(glf, self.w, self.h)
        prog.bind()
        # horizontal
        glf.glBindFramebuffer(0x8D40, tmp.fbo)
        glf.glViewport(0, 0, self.w, self.h)
        self._bind_raw(src.tex, 0)
        prog.setUniformValue("uTex", 0)
        # approximate gaussian radius via multi-tap offsets
        r = max(1.0, sigma_px)
        prog.setUniformValue("uDir", r * 1.0 / self.w, 0.0)
        self._draw_quad(prog, QRectF(0, 0, self.w, self.h), self.w, self.h)
        # vertical
        glf.glBindFramebuffer(0x8D40, dst.fbo)
        self._bind_raw(tmp.tex, 0)
        prog.setUniformValue("uDir", 0.0, r * 1.0 / self.h)
        self._draw_quad(prog, QRectF(0, 0, self.w, self.h), self.w, self.h)
        prog.release()

    def cleanup(self):
        if not self._glf:
            return
        glf = self._glf
        for f in (self.A, self.T, self.Bs, self.Bs2, self.Bb, self.Bb2, self.OUT):
            f.release(glf)
        from PySide6.QtOpenGL import QOpenGLTexture
        for t in list(self.mask_texes.values()):
            if isinstance(t, QOpenGLTexture):
                t.destroy()
            else:
                glf.glDeleteTextures(int(t))
        self.mask_texes.clear()
        for t in (self.base_tex, self.curve_tex, self.noise_tex):
            if isinstance(t, QOpenGLTexture):
                t.destroy()
            elif t:
                glf.glDeleteTextures(int(t))
        self.base_tex = self.curve_tex = self.noise_tex = None


def build_curve_texture(settings: dict, n=1024) -> np.ndarray:
    """RGBA LUT: R composite, G/B/A per-channel curves."""
    from .imaging import monotonic_spline
    xs = np.linspace(0, 1, n).astype(np.float32)
    lut = np.zeros((n, 4), dtype=np.uint8)

    def eval_pts(pts):
        if pts and len(pts) >= 2:
            _, ys = monotonic_spline(pts, n=n)
            return ys
        return xs.copy()

    comp = eval_pts(settings.get("curve_rgb"))
    rc = eval_pts(settings.get("curve_r"))
    gc = eval_pts(settings.get("curve_g"))
    bc = eval_pts(settings.get("curve_b"))
    # apply channel curve AFTER composite, matching CPU order
    for ch, cur in ((0, rc), (1, gc), (2, bc)):
        comp = np.interp(comp, xs, cur)
    lut[:, 0] = np.clip(comp * 255, 0, 255).astype(np.uint8)
    for ch, cur in ((1, rc), (2, gc), (3, bc)):
        lut[:, ch] = np.clip(cur * 255, 0, 255).astype(np.uint8)
    return lut
