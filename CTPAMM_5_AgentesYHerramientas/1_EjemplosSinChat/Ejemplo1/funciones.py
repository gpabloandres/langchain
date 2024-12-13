# Define una función para obtener la hora actual.
def get_current_time(*args, **kwargs):
    """Returns the current time in H:MM AM/PM format."""
    import datetime 
    now = datetime.datetime.now()
    return now.strftime("%I:%M %p")
