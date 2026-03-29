import asyncio
from typing import Optional
import xml.etree.ElementTree as ET

import numpy as np
import pyvips
import verovio

from lumachords.utils import Utils


class NotationRenderer:
    """
    Static (no instances) Verovio wrapper:
      - one shared toolkit
      - one async lock
      - atomic: setOptions -> loadData -> renderToSVG
    """

    MEI_UNIT_PX = 9.0 # half of staff-line distance in pixels (half-space)
    MEI_STAFF_LINE_COUNT = 5 # 5 staff lines
    MEI_STAFF_LINE_SPACE_UNITS = 2
    MEI_STAFF_HEIGHT_UNITS = MEI_STAFF_LINE_SPACE_UNITS * (MEI_STAFF_LINE_COUNT - 1)
    MEI_VERTICAL_EDGE_SPACING_UNITS = 6 # the spacing between (top edge-the first staff line) and (the last staff line-bottom edge)
    MEI_STAFF_SPACING_UNITS = 12 # min space between adjacent staves



    _tk: Optional["verovio.toolkit"] = None
    _lock: Optional[asyncio.Lock] = None

    @classmethod
    async def init(
        cls,
    ) -> None:
        """Static constructor. Call once (or safely many times) before rendering."""
        if cls._tk is not None:
            return

        cls._lock = asyncio.Lock()
        cls._tk = verovio.toolkit(initFont=False)
        if not cls._tk.setResourcePath(cls._tk.getResourcePath()):
            raise Exception("Verovio toolkit initialization failed: Data directory of Verovio does not exist.")

    @classmethod
    async def render_svg(
        cls,
        mei: str,
        options: dict,
        page: int = 1,
        max_page_count: int = 1,
    ) -> str:
        """Concurrency-safe render (works with different options per call)."""
        if cls._tk is None or cls._lock is None:
            await cls.init()

        assert cls._tk is not None
        assert cls._lock is not None

        async with cls._lock:
            if options is not None:
                cls._tk.resetOptions()
                cls._tk.setOptions(options)

            if not cls._tk.loadData(mei):
                raise RuntimeError(f"Verovio failed to load MEI data (len={len(mei)})")
            if max_page_count and cls._tk.getPageCount() > max_page_count:
                return None
            return cls._tk.renderToSVG(page)
    

    @classmethod
    async def svg_string_to_rgba(cls, svg: str, background_color = None, foreground_color: str=None, alpha_rate: float = 1.0, include_image_alpha_channel=True) -> np.ndarray:
        foreground_def = ""
        if foreground_color:
            foreground_def = f"""
    <style>
      svg * {{
        fill: {foreground_color} !important;
        stroke: {foreground_color} !important;
        color: {foreground_color} !important;
      }}
    </style>
    """
        
        bg_rect = f'<rect x="0" y="0" width="100%" height="100%" fill="{background_color}"/>\n' if background_color else ""

        style = foreground_def + bg_rect
        if len(style):
            # insert right after the opening <svg ...> tag
            insert_pos = svg.find('>') + 1
            svg = svg[:insert_pos] + style + svg[insert_pos:]

        v = pyvips.Image.svgload_buffer(svg.encode("utf-8"), dpi=72, scale=1.0)

        # Ensure 4 channels (RGBA)
        if v.bands == 3:
            v = v.bandjoin(255)  # add opaque alpha
        elif v.bands == 1:
            v = v.colourspace("srgb").bandjoin(255)

        mem = v.write_to_memory()
        rgba = np.frombuffer(mem, dtype=np.uint8).reshape(v.height, v.width, v.bands)
        img = rgba[..., [2, 1, 0, 3]]

        if include_image_alpha_channel:
            if alpha_rate < 1:
                img[:, :, 3] = np.round(img[:, :, 3] * alpha_rate).astype(img.dtype)
        else:
            img = img[:, :, :3]
        return img

    @staticmethod
    def calculate_scale(scaled_width_px, scaled_height_px, mei_width_px, mei_height_px, mei_margin_horizontal_px, mei_margin_vertical_px):
        scale_from_height = (
            scaled_height_px / (mei_height_px + 2 * mei_margin_vertical_px)
            if scaled_height_px
            else None
        )
        scale_from_width = (
            scaled_width_px / (mei_width_px + 2 * mei_margin_horizontal_px)
            if scaled_width_px
            else None
        )
        scale_candidates = [s for s in (scale_from_height, scale_from_width) if s]
        scale = min(scale_candidates) if scale_candidates else 1.0
        return scale


    @staticmethod
    def calculate_mei_units(scaled_width_px, scaled_height_px, margin_vertical_extra_units=None, margin_horizontal_extra_units=None):
        mei_height_units = (
            __class__.MEI_VERTICAL_EDGE_SPACING_UNITS +
            __class__.MEI_STAFF_HEIGHT_UNITS +
            __class__.MEI_STAFF_SPACING_UNITS +
            __class__.MEI_STAFF_HEIGHT_UNITS +
            __class__.MEI_VERTICAL_EDGE_SPACING_UNITS
        )
        mei_margin_vertical_units, mei_margin_horizontal_units = 4 + (margin_vertical_extra_units or 0), 1 + (margin_horizontal_extra_units or 0)

        loop_count = 2 if not (scaled_width_px and scaled_height_px) else 1

        for _ in range(loop_count):
            mei_height_px = mei_height_units * __class__.MEI_UNIT_PX
            mei_width_px = scaled_width_px if scaled_width_px else mei_height_px * 1.5
            mei_margin_horizontal_px = mei_margin_horizontal_units * __class__.MEI_UNIT_PX
            mei_margin_vertical_px = mei_margin_vertical_units * __class__.MEI_UNIT_PX

            scale = __class__.calculate_scale(scaled_width_px, scaled_height_px, mei_width_px, mei_height_px, mei_margin_horizontal_px, mei_margin_vertical_px)

            if not scaled_width_px:
                scaled_width_px = mei_width_px * scale
            if not scaled_height_px:
                scaled_height_px = mei_height_px * scale
        margin_vertical_px = int(max(0.0, (scaled_height_px / scale - mei_height_px) / 2))
        margin_horizontal_px = int(max(0.0, (scaled_width_px / scale - mei_width_px) / 2))
        margin_horizontal_px = int(min(mei_margin_horizontal_px, margin_horizontal_px) if margin_horizontal_px else mei_margin_horizontal_px)

        unscaled_width_px, unscaled_height_px = int(scaled_width_px / scale), int(scaled_height_px / scale)

        return {
            "scale": scale * 100,
            "pageWidth": unscaled_width_px,
            "pageHeight": unscaled_height_px,
            "pageMarginLeft": margin_horizontal_px,
            "pageMarginRight": margin_horizontal_px,
            "pageMarginTop": margin_vertical_px,
            "pageMarginBottom": margin_vertical_px,
        }


    async def verovio_svg_from_mei(
            mei: ET.Element,
            fixed_size: bool=False,
            print_measure_nums_interval: int=None,
            margin_vertical_extra_units=None,
            margin_horizontal_extra_units=None,
            output_width: int=None,
            output_height: int=None,
            max_page_count: int = 1,
        ) -> str:
        common_options = {
            "header": "none",
            "footer": "none",
            # CRITICAL: Force the last (or only) system to stretch to the full width
            "minLastJustification": 0.0,
        }
        if print_measure_nums_interval:
            common_options["mnumInterval"] = print_measure_nums_interval
        size_options = __class__.calculate_mei_units(output_width, output_height, margin_vertical_extra_units=margin_vertical_extra_units, margin_horizontal_extra_units=margin_horizontal_extra_units)
        if fixed_size:
            assert (output_width is not None or output_height is not None), "output_width or output_height must be specified when fixed_size=True"
            options = {                
                # Force Verovio to use this size (do NOT adjust/shrink to content)
                "adjustPageWidth": 0,
                "adjustPageHeight": 0,
            }
        else:
            options = {
                "breaks": "none",          # single system (no system/page breaks)  [oai_citation:3‡Reference book for Verovio](https://book.verovio.org/verovio-reference-book.pdf?utm_source=chatgpt.com)
                "adjustPageWidth": 1,      # crop width to content  [oai_citation:4‡Humdrum Plugin](https://plugin.humdrum.org/options/?utm_source=chatgpt.com)
                "adjustPageHeight": 1,     # crop height to content  [oai_citation:5‡Humdrum Plugin](https://plugin.humdrum.org/options/?utm_source=chatgpt.com)
            }
        options = {
            **common_options,
            **size_options,
            **options,
        }
        
        section_el = mei.find(".//section")
        while True:
            mei_str = Utils.xml_element_to_str(mei)
            svg = await NotationRenderer.render_svg(mei_str, options, max_page_count=max_page_count)
            if svg or section_el is None or not len(section_el):
                break
            for child in list(section_el):
                if child.tag == "measure" or child.tag.endswith("}measure"):
                    section_el.remove(child)
                    break

        return svg

