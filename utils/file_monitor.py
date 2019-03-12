import pyinotify
from datetime import datetime

path = '/tmp/audio.wav'
wm = pyinotify.WatchManager()  # Watch Manager
mask = pyinotify.IN_MODIFY  # watched events

class EventHandler(pyinotify.ProcessEvent):
    def startInference(self, event):
        print("[{}] New content appeared in -- {} file".format(datetime.now().isoformat(), event.pathname))
        """
        #### Do Not Block this function !!!
        Spawn an async process (or multiprocessing) for Inference processing.
        #### Do Not Block this function !!!
        """
        print("[{}] Done!".format(datetime.now().isoformat()))

    def process_IN_MODIFY(self, event):
        self.startInference(event)

handler = EventHandler()
notifier = pyinotify.Notifier(wm, handler)
wdd = wm.add_watch(path, mask, rec=True)

notifier.loop()
