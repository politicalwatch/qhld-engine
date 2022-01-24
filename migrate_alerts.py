from tipi_data.repositories.alerts import Alerts

alerts = Alerts.get_all()

for alert in alerts:
    searches = alert['searches']
    new_searches = []
    for search in searches:
        search_str = search['search']
        search_str = search_str.replace('tipiciudadano', 'politicas')
        search_str = search_str.replace('parlamento2030', 'ods')
        search['search'] = search_str 

    alert.save()
