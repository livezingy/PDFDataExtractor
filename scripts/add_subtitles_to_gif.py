"""
为GIF添加英文字幕工具
支持自动分析GIF帧并添加字幕，或使用配置文件手动指定字幕
"""
import sys
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from PIL import Image, ImageDraw, ImageFont
import json

try:
    import imageio
    IMAGEIO_AVAILABLE = True
except ImportError:
    IMAGEIO_AVAILABLE = False
    print("❌ imageio未安装，请运行: pip install imageio imageio-ffmpeg")

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
GIF_DIR = PROJECT_ROOT / "docs" / "screenshots" / "gifs"
CONFIG_DIR = PROJECT_ROOT / "docs" / "screenshots" / "gifs" / "subtitle_configs"


class GIFSubtitleAdder:
    """GIF字幕添加工具"""
    
    def __init__(self):
        self.font_cache = {}
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    
    def get_font(self, size: int = 24, bold: bool = False):
        """获取字体（使用微软雅黑）"""
        cache_key = (size, bold)
        if cache_key not in self.font_cache:
            try:
                # 尝试使用系统字体
                if sys.platform == "win32":
                    # 使用微软雅黑字体
                    font_path = "C:/Windows/Fonts/msyh.ttc"  # 微软雅黑
                    if bold:
                        font_path = "C:/Windows/Fonts/msyhbd.ttc"  # 微软雅黑粗体
                elif sys.platform == "darwin":
                    font_path = "/System/Library/Fonts/Helvetica.ttc"
                else:
                    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
                
                font = ImageFont.truetype(font_path, size)
            except:
                # 回退到默认字体
                try:
                    font = ImageFont.load_default()
                except:
                    font = None
            
            self.font_cache[cache_key] = font
        
        return self.font_cache[cache_key]
    
    def draw_subtitle(self, img: Image.Image, text: str, 
                     position: str = "bottom",
                     font_size: int = 24,
                     text_color: Tuple[int, int, int] = (255, 0, 0),
                     bg_color: Optional[Tuple[int, int, int]] = (0, 0, 0),
                     bg_alpha: int = 180,
                     padding: int = 10,
                     max_width: Optional[int] = None,
                     frame_number: Optional[int] = None) -> Image.Image:
        """
        在图片上绘制字幕（支持帧数显示）
        
        Args:
            img: 图片对象
            text: 字幕文本
            position: 字幕位置 ("top", "bottom", "center")
            font_size: 字体大小
            text_color: 文字颜色（默认红色）
            bg_color: 背景颜色
            bg_alpha: 背景透明度 (0-255)
            padding: 内边距
            max_width: 最大宽度（自动换行）
            frame_number: 帧数（如果提供，会在左侧显示）
        """
        draw = ImageDraw.Draw(img, 'RGBA')
        font = self.get_font(font_size)
        
        # 如果有帧数，格式化字幕文本：左侧帧数 | 右侧字幕
        if frame_number is not None:
            display_text = f"{frame_number} | {text}"
        else:
            display_text = text
        
        if not font:
            # 使用默认字体估算
            bbox = draw.textbbox((0, 0), display_text)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        else:
            bbox = draw.textbbox((0, 0), display_text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        
        # 处理文本换行
        if max_width and text_width > max_width:
            words = display_text.split()
            lines = []
            current_line = []
            current_width = 0
            
            for word in words:
                if font:
                    word_bbox = draw.textbbox((0, 0), word, font=font)
                    word_width = word_bbox[2] - word_bbox[0]
                else:
                    word_width = len(word) * font_size * 0.6
                
                if current_width + word_width > max_width and current_line:
                    lines.append(' '.join(current_line))
                    current_line = [word]
                    current_width = word_width
                else:
                    current_line.append(word)
                    current_width += word_width + (font_size * 0.3 if font else 10)
            
            if current_line:
                lines.append(' '.join(current_line))
            
            display_text = '\n'.join(lines)
            # 重新计算高度
            if font:
                bbox = draw.multiline_textbbox((0, 0), display_text, font=font)
            else:
                bbox = draw.textbbox((0, 0), display_text)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        
        img_width, img_height = img.size
        
        # 计算字幕位置（底端居中）
        if position == "top":
            y = padding
        elif position == "center":
            y = (img_height - text_height) // 2
        else:  # bottom
            y = img_height - text_height - padding
        
        x = (img_width - text_width) // 2
        
        # 绘制背景框
        if bg_color:
            bg_rect = [
                x - padding,
                y - padding,
                x + text_width + padding,
                y + text_height + padding
            ]
            draw.rectangle(bg_rect, fill=(*bg_color, bg_alpha))
        
        # 绘制文字
        if font:
            if '\n' in display_text:
                draw.multiline_text((x, y), display_text, fill=text_color, font=font, align='center')
            else:
                draw.text((x, y), display_text, fill=text_color, font=font)
        else:
            if '\n' in display_text:
                draw.multiline_text((x, y), display_text, fill=text_color, align='center')
            else:
                draw.text((x, y), display_text, fill=text_color)
        
        return img
    
    def analyze_gif(self, gif_path: Path) -> Dict:
        """
        分析GIF文件，获取帧信息
        
        Returns:
            包含帧数、时长等信息的字典
        """
        try:
            # 使用PIL读取GIF
            img = Image.open(gif_path)
            frame_count = 0
            durations = []
            
            try:
                while True:
                    frame_count += 1
                    if 'duration' in img.info:
                        durations.append(img.info['duration'] / 1000.0)  # 转换为秒
                    img.seek(img.tell() + 1)
            except EOFError:
                pass
            
            avg_duration = sum(durations) / len(durations) if durations else 0.1
            fps = 1.0 / avg_duration if avg_duration > 0 else 10.0
            total_duration = sum(durations) if durations else frame_count * avg_duration
            
            info = {
                'frame_count': frame_count,
                'duration': avg_duration,
                'fps': fps,
                'total_duration': total_duration
            }
            
            return info
        except Exception as e:
            print(f"❌ 分析GIF失败: {e}")
            # 尝试使用imageio
            if IMAGEIO_AVAILABLE:
                try:
                    reader = imageio.get_reader(gif_path)
                    metadata = reader.get_meta_data()
                    frame_count = 0
                    for _ in reader:
                        frame_count += 1
                    reader.close()
                    
                    duration = metadata.get('duration', 0.1)
                    fps = 1.0 / duration if duration > 0 else 10.0
                    
                    return {
                        'frame_count': frame_count,
                        'duration': duration,
                        'fps': fps,
                        'total_duration': frame_count * duration
                    }
                except:
                    pass
            return {}
    
    def add_subtitles(self, gif_path: Path, subtitle_config: List[Dict],
                     output_path: Optional[Path] = None,
                     position: str = "bottom",
                     font_size: int = 28,
                     text_color: Tuple[int, int, int] = (255, 255, 255),
                     bg_color: Optional[Tuple[int, int, int]] = (0, 0, 0),
                     bg_alpha: int = 200) -> bool:
        """
        为GIF添加字幕
        
        Args:
            gif_path: 输入GIF路径
            subtitle_config: 字幕配置列表，每个元素包含:
                - start_frame: 开始帧（从0开始）
                - end_frame: 结束帧（包含）
                - text: 字幕文本
            output_path: 输出路径（默认覆盖原文件）
            position: 字幕位置
            font_size: 字体大小
            text_color: 文字颜色
            bg_color: 背景颜色
            bg_alpha: 背景透明度
        """
        if not IMAGEIO_AVAILABLE:
            print("❌ imageio未安装")
            return False
        
        if output_path is None:
            output_path = gif_path.parent / f"{gif_path.stem}_subtitled{gif_path.suffix}"
        
        try:
            print(f"📖 读取GIF: {gif_path}")
            reader = imageio.get_reader(gif_path)
            metadata = reader.get_meta_data()
            # 设置每帧停留时间为1秒
            duration = 2.0
            
            frames = []
            frame_idx = 0
            
            print("🖼️  处理帧...")
            for frame in reader:
                img = Image.fromarray(frame)
                
                # 查找当前帧应该显示的字幕
                current_subtitle = None
                for subtitle in subtitle_config:
                    if subtitle['start_frame'] <= frame_idx <= subtitle['end_frame']:
                        current_subtitle = subtitle['text']
                        break
                
                # 如果有字幕，添加到图片上（传递帧数信息）
                if current_subtitle:
                    img = self.draw_subtitle(
                        img, current_subtitle,
                        position=position,
                        font_size=font_size,
                        text_color=text_color,
                        bg_color=bg_color,
                        bg_alpha=bg_alpha,
                        max_width=int(img.width * 0.8),
                        frame_number=frame_idx
                    )
                
                frames.append(img)
                frame_idx += 1
                
                if frame_idx % 10 == 0:
                    print(f"   已处理 {frame_idx} 帧...")
            
            reader.close()
            
            print(f"💾 保存带字幕的GIF: {output_path}")
            frames[0].save(
                output_path,
                save_all=True,
                append_images=frames[1:],
                duration=duration * 1000,  # 转换为毫秒（1秒 = 1000毫秒）
                loop=metadata.get('loop', 0)
            )
            
            print(f"✅ 完成！共处理 {len(frames)} 帧")
            return True
            
        except Exception as e:
            print(f"❌ 处理失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def generate_default_config(self, gif_path: Path) -> List[Dict]:
        """
        生成默认字幕配置（基于GIF分析）
        
        这是一个示例配置，用户需要根据实际内容调整
        """
        info = self.analyze_gif(gif_path)
        frame_count = info.get('frame_count', 100)
        
        if frame_count == 0:
            # 如果无法读取帧数，使用默认值
            frame_count = 100
        
        # 根据GIF时长自动分段
        total_duration = info.get('total_duration', 0)
        if total_duration > 0:
            # 每3-5秒一段字幕
            segment_duration = 4.0  # 每段4秒
            segments = max(1, int(total_duration / segment_duration))
        else:
            segments = 5
        
        frames_per_segment = max(1, frame_count // segments)
        
        # 为PDF Table Extractor创建默认字幕配置
        default_texts = [
            "PDF Table Extractor - Main Interface",
            "Upload PDF File and Configure Parameters",
            "Select Extraction Method and Flavor",
            "Processing and Detection",
            "View Extraction Results"
        ]
        
        config = []
        for i in range(segments):
            start = i * frames_per_segment
            end = (i + 1) * frames_per_segment - 1 if i < segments - 1 else frame_count - 1
            
            text = default_texts[i] if i < len(default_texts) else f"Scene {i + 1}"
            
            config.append({
                'start_frame': start,
                'end_frame': end,
                'text': text
            })
        
        return config
    
    def load_config(self, config_path: Path) -> Optional[List[Dict]]:
        """加载字幕配置"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ 加载配置失败: {e}")
            return None
    
    def save_config(self, config: List[Dict], config_path: Path):
        """保存字幕配置"""
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            print(f"✅ 配置已保存: {config_path}")
        except Exception as e:
            print(f"❌ 保存配置失败: {e}")


def main():
    """主函数"""
    # 使用PDFDataExtractorShow.gif文件
    gif_path = PROJECT_ROOT / "docs" / "screenshots" / "annotated" / "PDFDataExtractorShow.gif"
    
    if not gif_path.exists():
        print(f"❌ GIF文件不存在: {gif_path}")
        return
    
    adder = GIFSubtitleAdder()
    
    # 分析GIF
    print("=" * 60)
    print("GIF字幕添加工具")
    print("=" * 60)
    
    info = adder.analyze_gif(gif_path)
    print(f"\n📊 GIF信息:")
    print(f"   帧数: {info.get('frame_count', '未知')}")
    print(f"   帧率: {info.get('fps', 0):.2f} fps")
    print(f"   总时长: {info.get('total_duration', 0):.2f} 秒")
    
    # 检查是否有配置文件
    config_path = CONFIG_DIR / "PDFDataExtractorShow_subtitles.json"
    
    if config_path.exists():
        print(f"\n📝 加载配置文件: {config_path}")
        subtitle_config = adder.load_config(config_path)
        if subtitle_config:
            print(f"   找到 {len(subtitle_config)} 个字幕配置")
        else:
            print("   配置文件格式错误，使用默认配置")
            subtitle_config = adder.generate_default_config(gif_path)
            adder.save_config(subtitle_config, config_path)
    else:
        print(f"\n📝 生成默认配置文件: {config_path}")
        subtitle_config = adder.generate_default_config(gif_path)
        adder.save_config(subtitle_config, config_path)
        print("\n⚠️  请编辑配置文件，添加实际的字幕文本")
        print(f"   配置文件位置: {config_path}")
        print("\n配置格式示例:")
        print(json.dumps(subtitle_config, indent=2, ensure_ascii=False))
        return
    
    # 添加字幕（使用红色字体）
    print(f"\n🎬 开始添加字幕...")
    output_path = gif_path.parent / f"{gif_path.stem}_subtitled{gif_path.suffix}"
    success = adder.add_subtitles(
        gif_path,
        subtitle_config,
        output_path=output_path,
        position="bottom",
        font_size=28,
        text_color=(255, 0, 0),  # 红色字体
        bg_color=(0, 0, 0),
        bg_alpha=200
    )
    
    if success:
        print(f"\n✅ 带字幕的GIF已保存: {output_path}")
        print(f"\n💡 提示:")
        print(f"   - 如需调整字幕，请编辑: {config_path}")
        print(f"   - 然后重新运行此脚本")


if __name__ == "__main__":
    main()

