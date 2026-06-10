# monitoring.py
import time
from dataclasses import dataclass

@dataclass
class Event:
    name: str
    data: dict

class WallTimeHandler:
    def __call__(self, event, **data):
        if event.name == 'run_start':
            self.start = time.perf_counter_ns()
            self.wall_ns = 0
        if event.name == 'run_end':
            self.end = time.perf_counter_ns()
            self.wall_ns = self.end - self.start
            
class CPUTimeHandler:
    def __call__(self, event):
        if event.name == 'run_start':
            self.start = time.process_time_ns()
            self.cpu_ns = 0
        if event.name == 'run_end':
            self.end = time.process_time_ns()
            self.cpu_ns = self.end - self.start

class RecordHandler:
    def __init__(self):
        self.records = []
        
    def __call__(self, event):
        if event.name == 'run_end':
            self.records.append(event.data)
            
        
class Monitor:
    def __init__(self):
        self.handlers = []
        
    def subscribe(self, handler):
        self.handlers.append(handler)
        
    def emit(self, event):
        for handler in self.handlers:
            handler(event)