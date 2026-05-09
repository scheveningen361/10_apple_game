
import tkinter as tk

class AreaSelector:
    def __init__(self, master):
        self.master = master
        # Tkinter 윈도우 설정: 전체화면, 반투명
        self.master.attributes("-fullscreen", True)
        self.master.attributes("-alpha", 0.3)
        self.master.wait_visibility(self.master)
        self.master.configure(bg='grey')

        self.start_x = None
        self.start_y = None
        self.rect = None

        # 드래그 영역을 그릴 캔버스 생성
        self.canvas = tk.Canvas(self.master, cursor="cross", bg="grey", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # 마우스 이벤트 바인딩
        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)

    def on_button_press(self, event):
        """마우스 클릭 시 시작 좌표를 기록하고 사각형 그리기를 시작합니다."""
        self.start_x = self.canvas.canvasx(event.x)
        self.start_y = self.canvas.canvasy(event.y)

        if self.rect:
            self.canvas.delete(self.rect)
        self.rect = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline='red', width=2)

    def on_mouse_drag(self, event):
        """마우스 드래그 시 사각형의 크기를 실시간으로 변경하여 보여줍니다."""
        cur_x = self.canvas.canvasx(event.x)
        cur_y = self.canvas.canvasy(event.y)
        self.canvas.coords(self.rect, self.start_x, self.start_y, cur_x, cur_y)

    def on_button_release(self, event):
        """마우스 버튼에서 손을 떼면 최종 좌표를 계산하고 출력한 뒤 종료합니다."""
        end_x = self.canvas.canvasx(event.x)
        end_y = self.canvas.canvasy(event.y)
        
        # 시작점과 끝점에 상관없이 좌상단, 우하단 좌표를 계산
        x1 = min(self.start_x, end_x)
        y1 = min(self.start_y, end_y)
        x2 = max(self.start_x, end_x)
        y2 = max(self.start_y, end_y)

        print(f"Capture Area Coordinates: x={int(x1)}, y={int(y1)}, width={int(x2-x1)}, height={int(y2-y1)}")
        self.master.quit()

if __name__ == '__main__':
    root = tk.Tk()
    app = AreaSelector(root)
    root.mainloop()
    # mainloop가 끝나면 윈도우를 완전히 파괴
    root.destroy()
