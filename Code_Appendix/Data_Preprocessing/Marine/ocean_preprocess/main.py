import cdsapi
from datetime import datetime, timedelta
import util


def generate_month_dates(year, month):
    first_day = datetime(year, month, 1)

    if month == 12:
        next_month = datetime(year + 1, 1, 1)
    else:
        next_month = datetime(year, month + 1, 1)

    dates = []

    current_day = first_day
    while current_day < next_month:
        date_str = current_day.strftime('%d')
        dates.append(date_str)
        current_day += timedelta(days=1)

    return dates


util.build_folder_and_clean(f"./Data/")
c = cdsapi.Client()
for y in range(1900, 2010):
    for m in range(1, 13):
        d = generate_month_dates(y, m)

        config = {
            'variable': [
                'air_pressure_at_sea_level', 'air_temperature', 'dew_point_temperature',
                'water_temperature', 'wind_from_direction', 'wind_speed',
            ],
            'format': 'csv-obs.zip',
            'data_quality': ['passed', ],
            'year': f'{y}',
            'month': f'{m:02d}',
            'day': d,
        }

        c.retrieve(
            'insitu-observations-surface-marine',
            config, f'./Data/{y}-{m:02d}.zip'
        )
