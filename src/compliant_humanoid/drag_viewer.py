"""Mouse-only MuJoCo view: left drag applies a Cartesian perturbation."""
import glfw
import mujoco as mj
import numpy as np


class DragViewer:
    def __init__(self, model, data, key_callback):
        self.model, self.data, self.key_callback = model, data, key_callback
        self.cam = mj.MjvCamera()
        mj.mjv_defaultFreeCamera(model, self.cam)
        self.opt = mj.MjvOption()
        self.perturb = mj.MjvPerturb()
        self.scene = mj.MjvScene(model, maxgeom=2000)
        self.overlay = None
        self.last = (0., 0.)

    def __enter__(self):
        if not glfw.init():
            raise RuntimeError("GLFW could not open a desktop display")
        self.win = glfw.create_window(1100, 760, "G1 - left drag arm / right drag orbit / wheel zoom", None, None)
        if not self.win:
            glfw.terminate()
            raise RuntimeError("Could not create MuJoCo window")
        glfw.make_context_current(self.win)
        self.ctx = mj.MjrContext(self.model, mj.mjtFontScale.mjFONTSCALE_150)
        glfw.set_mouse_button_callback(self.win, self.button)
        glfw.set_cursor_pos_callback(self.win, self.move)
        glfw.set_scroll_callback(self.win, lambda w,x,y: setattr(self.cam, "distance",
                                 np.clip(self.cam.distance*np.exp(-.12*y), .25, 12)))
        glfw.set_key_callback(self.win, lambda w,k,s,a,m: self.key_callback(k) if a == glfw.PRESS else None)
        glfw.set_window_focus_callback(self.win, lambda w,focused: self.release() if not focused else None)
        return self

    def release(self):
        self.perturb.active = 0

    def button(self, win, button, action, mods):
        self.last = x,y = glfw.get_cursor_pos(win)
        if button != glfw.MOUSE_BUTTON_LEFT:
            return
        if action == glfw.RELEASE:
            self.release()
            return
        width,height = glfw.get_window_size(win)
        point = np.zeros(3)
        ids = [np.zeros(1, dtype=np.int32) for _ in range(3)]
        body = mj.mjv_select(self.model, self.data, self.opt, width/height, x/width,
                             1-y/height, self.scene, point, *ids)
        if body <= 0 or not self.model.body(body).name.startswith(
                ("left_shoulder", "left_elbow", "left_wrist", "left_rubber_hand")):
            return
        self.perturb.select = body
        self.perturb.localpos[:] = self.data.xmat[body].reshape(3,3).T @ (point-self.data.xpos[body])
        mj.mjv_initPerturb(self.model, self.data, self.scene, self.perturb)
        self.perturb.active = mj.mjtPertBit.mjPERT_TRANSLATE

    def move(self, win, x, y):
        dx,dy = np.subtract((x,y), self.last)/max(1, glfw.get_window_size(win)[1])
        self.last = x,y
        if self.perturb.active:
            mj.mjv_movePerturb(self.model, self.data, mj.mjtMouse.mjMOUSE_MOVE_V,
                               dx, dy, self.scene, self.perturb)
        elif glfw.get_mouse_button(win, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS:
            mj.mjv_moveCamera(self.model, mj.mjtMouse.mjMOUSE_ROTATE_H, dx,dy,self.scene,self.cam)

    def sync(self):
        glfw.poll_events()
        self.data.xfrc_applied[:] = 0
        mj.mjv_applyPerturbForce(self.model, self.data, self.perturb)
        mj.mjv_updateScene(self.model,self.data,self.opt,self.perturb,self.cam,
                           mj.mjtCatBit.mjCAT_ALL,self.scene)
        rect = mj.MjrRect(0,0,*glfw.get_framebuffer_size(self.win))
        mj.mjr_render(rect,self.scene,self.ctx)
        if self.overlay:
            _,pos,labels,values = self.overlay
            mj.mjr_overlay(mj.mjtFont.mjFONT_NORMAL,pos,rect,labels,values,self.ctx)
        glfw.swap_buffers(self.win)

    def set_texts(self, overlay):
        self.overlay = overlay

    def is_running(self):
        return not glfw.window_should_close(self.win)

    def __exit__(self, *args):
        self.ctx.free()
        glfw.destroy_window(self.win)
        glfw.terminate()
