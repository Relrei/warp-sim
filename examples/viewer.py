"""
Real-time PBF fluid viewer — GLFW + OpenGL point sprites.
Works on Wayland (Hyprland) and X11.

Controls:
  Left drag   : rotate camera (orbit)
  Scroll      : zoom
  Space       : pause / resume
  R           : reset simulation
  Q / Esc     : quit

Usage:
  python examples/viewer.py
  python examples/viewer.py --particles 500000 --h 0.12
"""

import sys, argparse, math, ctypes
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

import numpy as np
import glfw
from OpenGL import GL
from OpenGL.GL import *
from OpenGL.GL.shaders import compileShader, compileProgram
import warp as wp

from sim.scene      import SimConfig, Domain
from sim.solver_pbf import PBFSolver


# ---------------------------------------------------------------------------
# Minimal shaders — point sprites colored by depth
# ---------------------------------------------------------------------------
VERT = """
#version 330 core
layout(location = 0) in vec3 pos;
uniform mat4 mvp;
uniform float point_size;
void main() {
    gl_Position  = mvp * vec4(pos, 1.0);
    gl_PointSize = point_size / gl_Position.w;
}
"""

FRAG = """
#version 330 core
out vec4 color;
void main() {
    vec2 c = gl_PointCoord - 0.5;
    if (dot(c, c) > 0.25) discard;
    float depth = 1.0 - gl_FragCoord.z;
    color = vec4(0.1 + depth*0.3, 0.4 + depth*0.3, 0.9, 1.0);
}
"""


def _link_program(vert_src, frag_src):
    return compileProgram(
        compileShader(vert_src, GL_VERTEX_SHADER),
        compileShader(frag_src, GL_FRAGMENT_SHADER),
    )


def _perspective(fov_deg, aspect, near, far):
    f  = 1.0 / math.tan(math.radians(fov_deg) / 2)
    nf = 1.0 / (near - far)
    return np.array([
        [f/aspect, 0,  0,                    0],
        [0,        f,  0,                    0],
        [0,        0,  (far+near)*nf,        -1],
        [0,        0,  2*far*near*nf,         0],
    ], dtype=np.float32)

def _lookat(eye, center, up):
    f = np.array(center) - np.array(eye); f /= np.linalg.norm(f)
    r = np.cross(f, np.array(up));       r /= np.linalg.norm(r)
    u = np.cross(r, f)
    M = np.eye(4, dtype=np.float32)
    M[0,:3]=r; M[1,:3]=u; M[2,:3]=-f
    T = np.eye(4, dtype=np.float32)
    T[:3,3] = -np.array(eye)
    return M @ T.T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--particles", type=int,   default=500_000)
    ap.add_argument("--h",         type=float, default=0.12)
    ap.add_argument("--substeps",  type=int,   default=6)
    ap.add_argument("--width",     type=int,   default=1280)
    ap.add_argument("--height",    type=int,   default=720)
    args = ap.parse_args()

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------
    cfg = SimConfig(
        domain           = Domain(size=(8.0, 4.0, 8.0)),
        max_particles    = args.particles,
        smoothing_length = args.h,
        substeps         = args.substeps,
        fps              = 60.0,
        gravity          = -9.8,
    )
    solver = PBFSolver(cfg)
    solver.reset_dam(fill=(0.4, 1.0, 0.8))

    # ------------------------------------------------------------------
    # GLFW window
    # ------------------------------------------------------------------
    if not glfw.init():
        raise RuntimeError("GLFW init failed")

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

    win = glfw.create_window(args.width, args.height,
                             f"warp-sim  [{args.particles:,} particles]",
                             None, None)
    if not win:
        glfw.terminate(); raise RuntimeError("Window creation failed")

    glfw.make_context_current(win)
    glfw.swap_interval(1)   # vsync

    # ------------------------------------------------------------------
    # OpenGL setup
    # ------------------------------------------------------------------
    prog   = _link_program(VERT, FRAG)
    u_mvp  = glGetUniformLocation(prog, "mvp")
    u_size = glGetUniformLocation(prog, "point_size")

    vao = glGenVertexArrays(1)
    vbo = glGenBuffers(1)
    glBindVertexArray(vao)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, args.particles * 12, None, GL_DYNAMIC_DRAW)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)
    glBindVertexArray(0)

    glEnable(GL_DEPTH_TEST)
    glEnable(GL_PROGRAM_POINT_SIZE)
    glClearColor(0.05, 0.05, 0.08, 1.0)

    # ------------------------------------------------------------------
    # Camera (orbit)
    # ------------------------------------------------------------------
    cx, cy, cz = cfg.domain.x/2, cfg.domain.z/2, cfg.domain.y/2
    center  = np.array([cx, cy, cz], np.float32)
    radius  = [20.0]
    yaw     = [math.pi * 1.1]
    pitch   = [0.35]
    mouse   = {"drag": False, "x": 0.0, "y": 0.0}
    paused  = [False]

    def eye_pos():
        e = center + radius[0] * np.array([
            math.cos(pitch[0]) * math.sin(yaw[0]),
            math.sin(pitch[0]),
            math.cos(pitch[0]) * math.cos(yaw[0]),
        ])
        return e

    def on_scroll(_, __, dy):
        radius[0] = max(2.0, radius[0] - dy * 1.2)

    def on_mouse_button(_, btn, action, __):
        if btn == glfw.MOUSE_BUTTON_LEFT:
            mouse["drag"] = (action == glfw.PRESS)
            if action == glfw.PRESS:
                x, y = glfw.get_cursor_pos(_)
                mouse["x"], mouse["y"] = x, y

    def on_cursor(_, x, y):
        if mouse["drag"]:
            dx = x - mouse["x"]; dy = y - mouse["y"]
            yaw[0]   -= dx * 0.005
            pitch[0]  = max(-1.4, min(1.4, pitch[0] - dy * 0.005))
            mouse["x"], mouse["y"] = x, y

    def on_key(_, key, sc, action, mods):
        if action == glfw.PRESS:
            if key in (glfw.KEY_Q, glfw.KEY_ESCAPE):
                glfw.set_window_should_close(win, True)
            elif key == glfw.KEY_SPACE:
                paused[0] = not paused[0]
                print("Paused" if paused[0] else "Running")
            elif key == glfw.KEY_R:
                solver.reset_dam(fill=(0.4, 1.0, 0.8))
                print("Reset")

    glfw.set_scroll_callback(win, on_scroll)
    glfw.set_mouse_button_callback(win, on_mouse_button)
    glfw.set_cursor_pos_callback(win, on_cursor)
    glfw.set_key_callback(win, on_key)

    print(f"\n[viewer] {args.particles:,} particles  h={args.h}")
    print("[viewer] Left drag=rotate  Scroll=zoom  Space=pause  R=reset  Q=quit\n")

    pt_size = max(4.0, args.h * 80.0)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    while not glfw.window_should_close(win):
        glfw.poll_events()

        if not paused[0]:
            solver.step()

        # Upload particle positions to VBO
        pos_np = solver.p.numpy().astype(np.float32)   # (N, 3)
        flags  = solver.flag.numpy()
        live   = pos_np[flags == 1]
        n_live = len(live)

        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferSubData(GL_ARRAY_BUFFER, 0, live.nbytes, live)

        # Render
        w, h = glfw.get_framebuffer_size(win)
        glViewport(0, 0, w, h)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        proj = _perspective(45.0, w / max(h, 1), 0.1, 500.0)
        view = _lookat(eye_pos(), center, [0, 1, 0])
        mvp  = (proj @ view).astype(np.float32)

        glUseProgram(prog)
        glUniformMatrix4fv(u_mvp, 1, GL_FALSE, mvp)
        glUniform1f(u_size, pt_size)
        glBindVertexArray(vao)
        glDrawArrays(GL_POINTS, 0, n_live)
        glBindVertexArray(0)

        glfw.swap_buffers(win)

    glfw.terminate()
    print("[viewer] closed")


if __name__ == "__main__":
    main()
