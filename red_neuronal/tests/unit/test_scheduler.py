from django.test import TestCase
from apscheduler.schedulers.background import BackgroundScheduler
import time
from datetime import datetime, timedelta


class SchedulerTestCase(TestCase):
    def test_scheduler_job(self):
        scheduler = BackgroundScheduler()
        scheduler.start()
        job_executed = {"executed": False}

        def my_task():
            job_executed["executed"] = True

        run_time = datetime.now() + timedelta(seconds=2)
        scheduler.add_job(my_task, 'date', run_date=run_time)
        time.sleep(3)
        self.assertTrue(job_executed["executed"])
        scheduler.shutdown()
