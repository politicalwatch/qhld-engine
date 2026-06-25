import tipi_tasks


class SendAlerts():
    def __init__(self):
        '''
        Add init() before send_alerts() to ensure it always use the same
        celery instance, despite flask multithrading
        '''
        tipi_tasks.init()
        tipi_tasks.alerts.send_alerts.apply_async()


if __name__ == "__main__":
    SendAlerts()
