import time

class TimeKeeper():
    def __init__(self):
        self.current_time = time.time()

    def update_time(self):
        self.current_time = time.time()
    
    def set_time(self, customized_time):
        self.current_time = customized_time

    def get_time_current(self):
        return self.current_time
    
    def get_time_elapse(self):
        return time.time() - self.current_time

    def get_update_time(self, round_digit = 2):
        time_elapse = self.get_time_elapse()
        self.update_time()
        return round(time_elapse, 2)