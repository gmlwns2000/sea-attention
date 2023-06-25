events ={} # "plot_perlin_attention_called":[plot_perlin_attention()]

def register_event(event: str, function: callable):
    handlers = events.get(event)
    
    if handlers is None:
        events[event] = [function]
    else:
        handlers.append(function)

def dispatch(event: str, current_state): # **kwargs
    handlers = events.get(event)
    
    if event is None:
        raise ValueError(f"Event {event} was not found")
    
    for handler in handlers:
        handler(current_state)