import cv2
import numpy as np
import pygame
from OpenGL.GL import *

from .window_geometry_helper import Panel, TitlePosition, Viewport, WindowDef


class WindowGLHelper:
    @staticmethod
    def begin_frame():
        glClear(GL_COLOR_BUFFER_BIT)

    @staticmethod
    def restore_viewport(wdef: WindowDef):
        # Restore viewport to full window
        glViewport(0, 0, *wdef.window_size)

    @staticmethod
    def begin_gl_context(wdef: WindowDef):
        glMatrixMode(GL_PROJECTION); glPushMatrix(); glLoadIdentity()
        glOrtho(0, wdef.window_size[0], 0, wdef.window_size[1], -1, 1)
        glMatrixMode(GL_MODELVIEW);  glPushMatrix(); glLoadIdentity()

    @staticmethod
    def end_gl_context():
        # Restore previous matrices
        glMatrixMode(GL_MODELVIEW);  glPopMatrix()
        glMatrixMode(GL_PROJECTION); glPopMatrix()

    @staticmethod
    def begin_gl_blend():
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    @staticmethod
    def end_gl_blend():
        glDisable(GL_BLEND)
        glColor4f(1,1,1,1)

    @staticmethod
    def reset_gl_state():
        glDisable(GL_DEPTH_TEST)
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)

    @staticmethod
    def allocate_panel_textures(panels: dict[int, Panel]) -> None:
        """Allocate image textures for all panels using their viewport sizes."""
        for panel in panels.values():
            if not panel.viewport or panel.has_variable_size:
                continue
            w, h = panel.viewport.w, panel.viewport.h

            tex_id = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, tex_id)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            # Pre-allocate size only; data will be uploaded later.
            glTexImage2D(
                GL_TEXTURE_2D,
                0,
                GL_RGB8,
                w,
                h,
                0,
                GL_BGR,
                GL_UNSIGNED_BYTE,
                None,
            )
            glBindTexture(GL_TEXTURE_2D, 0)

            panel.image_tex = int(tex_id)

    @staticmethod
    def delete_panel_textures(panels: dict[int, Panel]) -> None:
        """Delete image textures for all panels."""
        for panel in panels.values():
            if panel.image_tex:
                glDeleteTextures(int(panel.image_tex))
            if panel.title_tex:
                glDeleteTextures(int(panel.title_tex))

    @staticmethod
    def cleanup_gl():
        glBindTexture(GL_TEXTURE_2D, 0)
        glFlush(); glFinish()

    @staticmethod
    def create_title_textures(panels: dict[int, Panel], font: pygame.font.Font) -> None:
        """Create textures for panel titles."""
        for panel in panels.values():
            if not (panel.title and len(panel.title)):
                continue

            surf = font.render(panel.title, True, (255, 255, 255))
            label = pygame.Surface(surf.get_size(), pygame.SRCALPHA)
            label.blit(surf, (0, 0))
            w, h = label.get_width(), label.get_height()
            buf = pygame.image.tostring(label, "RGBA", False)

            tex_id = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, tex_id)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexImage2D(
                GL_TEXTURE_2D,
                0,
                GL_RGBA8,
                w,
                h,
                0,
                GL_RGBA,
                GL_UNSIGNED_BYTE,
                buf,
            )
            glBindTexture(GL_TEXTURE_2D, 0)

            panel.title_tex = (int(tex_id), w, h)

    @staticmethod
    def update_panel_image(panel: Panel, frame_bgr: np.ndarray) -> None:
        """
        Copy a BGR image into a pre-allocated panel texture.
        Expected shape: (h, w, 3), dtype=uint8.
        """
        if panel is None or panel.viewport is None:
            return

        w, h = panel.viewport.w, panel.viewport.h
        if frame_bgr is None or frame_bgr.size == 0:
            return
        if panel.has_variable_size:
            # If the layout provides a size, prefer it; otherwise use image size.
            if w is None or h is None or w <= 0 or h <= 0:
                h, w = frame_bgr.shape[:2]
        else:
            if w is None or h is None or w <= 0 or h <= 0:
                h, w = frame_bgr.shape[:2]

        if panel.has_variable_size:
            panel.update_size(w, h)
            if panel.image_tex:
                tex_id, _, _ = panel.image_tex
                glDeleteTextures(int(tex_id))
            tex_id = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, tex_id)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexImage2D(
                GL_TEXTURE_2D,
                0,
                GL_RGB8,
                w,
                h,
                0,
                GL_BGR,
                GL_UNSIGNED_BYTE,
                None,
            )
            panel.image_tex = (int(tex_id), w, h)
            glBindTexture(GL_TEXTURE_2D, 0)
        else:
            tex_id = panel.image_tex

        # Resize input image to match the fixed texture size if needed
        if (frame_bgr.shape[1] != w or frame_bgr.shape[0] != h) and (w > 0 and h > 0):
            frame_bgr = cv2.resize(frame_bgr, (w, h), interpolation=cv2.INTER_AREA)

        glBindTexture(GL_TEXTURE_2D, tex_id)
        glTexSubImage2D(
            GL_TEXTURE_2D,
            0,
            0,
            0,
            w,
            h,
            GL_BGR,
            GL_UNSIGNED_BYTE,
            frame_bgr,
        )
        glBindTexture(GL_TEXTURE_2D, 0)

    @staticmethod
    def draw_panel_images(panels: dict[int, Panel]) -> None:
        """Draw all image textures in their panel slots."""
        __class__.begin_gl_blend()
        glEnable(GL_TEXTURE_2D)
        for panel in panels.values():
            if panel.image_tex is None or not panel.viewport:
                continue

            x, y, w, h = panel.viewport.as_tuple()
            if isinstance(panel.image_tex, tuple):
                tex_id, w, h = panel.image_tex
            else:
                tex_id = panel.image_tex
            glViewport(x, y, w, h)

            glBindTexture(GL_TEXTURE_2D, tex_id)
            glBegin(GL_QUADS)
            # Flip vertically so top-left origin images render upright in GL.
            glTexCoord2f(0.0, 1.0); glVertex2f(-1.0, -1.0)
            glTexCoord2f(1.0, 1.0); glVertex2f(1.0, -1.0)
            glTexCoord2f(1.0, 0.0); glVertex2f(1.0, 1.0)
            glTexCoord2f(0.0, 0.0); glVertex2f(-1.0, 1.0)
            glEnd()
            glBindTexture(GL_TEXTURE_2D, 0)

        glDisable(GL_TEXTURE_2D)
        __class__.end_gl_blend()

    @staticmethod
    def draw_panel_titles(panels: dict[int, Panel]) -> None:
        __class__.begin_gl_blend()
        glEnable(GL_TEXTURE_2D)

        for panel in panels.values():
            if not panel.viewport or not panel.title_tex:
                continue

            tex, tw, th = panel.title_tex
            x, y, w, h = panel.viewport.as_tuple()

            if panel.title_pos == TitlePosition.INSIDE_TOP_LEFT:
                px = x + 10
                py = y + h + 4 - int(th * 2)
            else: # TitlePosition.OUTSIDE_TOP_CENTER
                # Simple placement: centered horizontally, just above the panel
                px = x + (w - tw) // 2
                py = y + h + 4

            glBindTexture(GL_TEXTURE_2D, tex)
            glBegin(GL_QUADS)
            # Flip vertically via texcoords (no software flip on upload)
            glTexCoord2f(0.0, 1.0); glVertex2f(px,       py)
            glTexCoord2f(1.0, 1.0); glVertex2f(px + tw,  py)
            glTexCoord2f(1.0, 0.0); glVertex2f(px + tw,  py + th)
            glTexCoord2f(0.0, 0.0); glVertex2f(px,       py + th)
            glEnd()
            glBindTexture(GL_TEXTURE_2D, 0)

        glDisable(GL_TEXTURE_2D)
        __class__.end_gl_blend()

    @staticmethod
    def _rgba255(c):
        # accept 0..1 or 0..255; return 0..1
        if isinstance(c, (list, tuple)) and len(c) in (3,4):
            if max(c) > 1.0:
                return tuple([v/255.0 for v in (c if len(c)==4 else (*c,255))])
            return tuple(c if len(c)==4 else (*c,1.0))
        return (1,1,1,1)

    @staticmethod
    def build_surface_texture(surf_rgba: pygame.Surface, tw: int=None, th: int=None, existing_tex_id: int=None) -> tuple[int, int, int]:
        if existing_tex_id:
            if isinstance(existing_tex_id, tuple):
                existing_tex_id, _, _ = existing_tex_id
            if existing_tex_id:
                glDeleteTextures(int(existing_tex_id))
        buf = pygame.image.tostring(surf_rgba, "RGBA", False)
        surf_tw, surf_th = surf_rgba.get_width(), surf_rgba.get_height()
        tw = tw or surf_tw
        th = th or surf_th

        tex = glGenTextures(1)
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, tex)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, tw, th, 0, GL_RGBA, GL_UNSIGNED_BYTE, buf)
        glBindTexture(GL_TEXTURE_2D, 0)
        glDisable(GL_TEXTURE_2D)
        return int(tex), tw, th

    @staticmethod
    def draw_surface(surf_rgba: pygame.Surface, tx: int, ty: int, tw: int=None, th: int=None):
        tex, tw, th = __class__.build_surface_texture(surf_rgba, tw=tw, th=th)

        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, tex)
        glBegin(GL_QUADS)
        # Use flipped V so text isn't mirrored
        glTexCoord2f(0, 1); glVertex2f(tx,     ty)
        glTexCoord2f(1, 1); glVertex2f(tx+tw,  ty)
        glTexCoord2f(1, 0); glVertex2f(tx+tw,  ty+th)
        glTexCoord2f(0, 0); glVertex2f(tx,     ty+th)
        glEnd()

        glBindTexture(GL_TEXTURE_2D, 0)
        glDeleteTextures(int(tex))  # simple; for perf, cache by text string
        glDisable(GL_TEXTURE_2D)

    @staticmethod
    def draw_progress_bar(viewport: Viewport,
                        font: pygame.font.Font,
                        progress:float,
                        text:str|None=None,
                        bg=(32,32,32,220),
                        fg=(60,180,90,255),
                        border=(255,255,255,128),
                        text_color=(255,255,255,255)):
        """
        Draw a progress bar at pixel rect (x,y,w,h) with origin at bottom-left.
        Call from your render path (e.g., at end of gl_present()).
        """
        # Clamp between 0..1
        p = 0.0 if progress < 0 else 1.0 if progress > 1 else float(progress)
        bg = __class__._rgba255(bg); fg = __class__._rgba255(fg); border = __class__._rgba255(border); text_color = __class__._rgba255(text_color)

        x, y, w, h = viewport.as_tuple()

        __class__.begin_gl_blend()

        # Background
        glDisable(GL_TEXTURE_2D)
        glColor4f(*bg)
        glBegin(GL_QUADS)
        glVertex2f(x,     y)
        glVertex2f(x+w,   y)
        glVertex2f(x+w,   y+h)
        glVertex2f(x,     y+h)
        glEnd()

        # Fill
        fw = int(w * p)
        if fw > 0:
            glColor4f(*fg)
            glBegin(GL_QUADS)
            glVertex2f(x,     y)
            glVertex2f(x+fw,  y)
            glVertex2f(x+fw,  y+h)
            glVertex2f(x,     y+h)
            glEnd()

        # Border
        glColor4f(*border)
        glLineWidth(1.5)
        glBegin(GL_LINE_LOOP)
        glVertex2f(x,     y)
        glVertex2f(x+w,   y)
        glVertex2f(x+w,   y+h)
        glVertex2f(x,     y+h)
        glEnd()

        # Text (centered)
        if text:
            surf = font.render(text, True,
                                        tuple(int(255*c) for c in text_color[:3]))
            tw, th = surf.get_width(), surf.get_height()
            surf_rgba = pygame.Surface((tw, th), pygame.SRCALPHA, 32)
            surf_rgba.blit(surf, (0, 0))
            tx = int(x + (w - tw) * 0.5)
            ty = int(y + (h - th) * 0.5)
            __class__.draw_surface(surf_rgba, tx, ty)

        # Restore state
        glDisable(GL_TEXTURE_2D)
        __class__.end_gl_blend()

    @staticmethod
    def render_multiline_text(text, font, color, aa=True, bg=None, align="left", line_spacing=1, fixed_line_count=None, min_line_count=None):
        """
        Render a text string that may contain '\n' into a single Surface.
        align: 'left' | 'center' | 'right'
        """
        if not text:
            text = ""
        lines = text.splitlines() or [""]
        if fixed_line_count:
            if min_line_count and fixed_line_count < min_line_count:
                fixed_line_count = min_line_count
            if len(lines) > fixed_line_count:
                lines = lines[:fixed_line_count]
            elif len(lines) < fixed_line_count:
                lines += [""] * (fixed_line_count - len(lines))
        elif min_line_count and len(lines) < min_line_count:
            lines += [""] * (min_line_count - len(lines))

        line_surfs = []
        max_w = 0
        line_h = font.get_linesize() * 1.5

        for line in lines:
            # Render each line; keep blank lines as empty height
            surf = font.render(line, aa, color, bg)
            line_surfs.append(surf)
            if surf.get_width() > max_w:
                max_w = surf.get_width()

        total_h = int(len(lines) * line_h * line_spacing)
        # Use SRCALPHA for transparent background
        out = pygame.Surface((max_w, total_h), pygame.SRCALPHA, 32)
        if bg is not None:
            out.fill(bg)

        y = 0
        for s in line_surfs:
            if align == "center":
                x = (max_w - s.get_width()) // 2
            elif align == "right":
                x = max_w - s.get_width()
            else:
                x = 0
            # Vertically place each line on the line grid (using linesize)
            out.blit(s, (x, y + (line_h - s.get_height()) // 2))
            y += line_h * line_spacing

        tw, th = max_w, total_h
        return out, tw, th
