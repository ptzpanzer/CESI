import os
import ftplib
import datetime
import time


def download_task(task_name, link, ld):
    file_name = task_name.split('/')[-1]
    print(f"Working on {file_name}")

    try:
        with open(os.path.join(ld, file_name), 'wb') as local_file:
            link.retrbinary('RETR ' + task_name, local_file.write)
        return 1
    except Exception as e:
        if "550" in str(e):
            print(f"\tAssuming file not exist: {e}, skipping.")
            with open('./err.log', 'a') as flog:
                flog.write(f'Assuming file {task} not exist.\n')
            return 0
        else:
            print(f"\tFTP error occurred: {e}, Reconnecting after 60s...")
            return -1


def date_range(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + datetime.timedelta(n)


if __name__ == '__main__':
    start_date = datetime.date(2001, 7, 1)
    end_date = datetime.date(2004, 3, 31)
    ftp_server = 'madis-data.ncep.noaa.gov'
    username = 'anonymous'
    password = 'anonymous'
    remote_dir = '/archive'
    local_dir = './downloaded_files'

    # set local env and ftp
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)
    ftp = ftplib.FTP(ftp_server)
    ftp.login(username, password)

    # construct task list
    tasks = []
    for current_date in date_range(start_date, end_date):
        for hour in range(24):
            tasks.append(
                f"{remote_dir}/{current_date.year}/{current_date.month:02d}/{current_date.day:02d}/point/acars/netcdf/{current_date.year}{current_date.month:02d}{current_date.day:02d}_{hour:02d}00.gz"
            )
    print(f"Total Tasks：{len(tasks)}")

    # check finished tasks
    local_files = os.listdir(local_dir)
    tasks_to_remove = []
    for task in tasks:
        filename = task.split('/')[-1]
        if filename in local_files:
            tasks_to_remove.append(task)
    print(f"Downloaded：{len(tasks_to_remove)}")

    # renew task list
    for task in tasks_to_remove:
        tasks.remove(task)
    print(f"Actual Processed：{len(tasks)}")

    results = []
    for task in tasks:
        while True:
            value = download_task(task, ftp, local_dir)

            if value == 0 or value == 1:
                break
            elif value == -1:
                time.sleep(60)
                ftp = ftplib.FTP(ftp_server)
                ftp.login(username, password)

    successful_downloads = sum(results)
    print(f"Successfully downloaded {successful_downloads} files.")
