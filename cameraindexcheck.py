from pygrabber.dshow_graph import FilterGraph

def list_cameras():
    graph = FilterGraph()
    devices = graph.get_input_devices()
    for idx, name in enumerate(devices):
        print(f"Index: {idx} | Nama Kamera: {name}")

if __name__ == "__main__":
    list_cameras()
