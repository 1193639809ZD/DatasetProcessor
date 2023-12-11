import time

import requests

from aiearth.openapi import build_client
from aiearth.openapi.enums import *
from aiearth.openapi.models import *

CLIENT = build_client('LTAI5tRx18U1yg4GBi5qqHqw', 'HVvaleNx5DKwWfL5ahO7bZob6pf6Be')


def try_enum(enum_class, enum_value):
    try:
        return enum_class(enum_value)
    except Exception as e:
        return enum_value


def sync_pulish_raster(name: str, download_url: str) -> (str, PublishStatus):
    """
    调用发布办法，并查询发布状态，直到发布完成或者发布失败
    :param name: 影像名称
    :param download_url: 影像下载地址
    :return: (影像ID， 影像发布状态)
    """
    publish_raster_req: PublishRasterRequest = PublishRasterRequest()
    publish_raster_req.name = name
    publish_raster_req.download_url = download_url
    publish_raster_req.file_type = RasterFileType.TIFF.value

    publish_raster_resp: PublishRasterResponse = CLIENT.publish_raster(publish_raster_req)

    data_id = publish_raster_resp.body.data_id
    publish_status = None
    while True:
        list_user_raster_datas_req: ListUserRasterDatasRequest = ListUserRasterDatasRequest()
        list_user_raster_datas_req.data_id = data_id
        list_user_raster_datas_req.from_type = UserDataFromType.PERSONAL.value
        list_user_raster_datas_req.page_number = 1
        list_user_raster_datas_req.page_size = 1
        list_user_raster_datas_resp: ListUserRasterDatasResponse = CLIENT.list_user_raster_datas(
            list_user_raster_datas_req)

        name = list_user_raster_datas_resp.body.list[0].raster.name
        status = list_user_raster_datas_resp.body.list[0].raster.publish_status
        print(f"DataName: {name}, Status: {status}")
        if status == PublishStatus.PUBLISHDONE.value:
            publish_status = PublishStatus.PUBLISHDONE
            break
        elif status == PublishStatus.PUBLISHFAIL.value:
            print(list_user_raster_datas_resp.body.list[0].raster.publish_msg)
            publish_status = PublishStatus.PUBLISHFAIL
            break
        else:
            time.sleep(3)
    return data_id, publish_status


def sync_create_lcc_job(data_id: str) -> (JobStatus, str):
    createAiJobReq: CreateAIJobRequest = CreateAIJobRequest()
    createAiJobReq.job_name = 'openapi_create_landclf_job'
    createAiJobReqInputSrc = CreateAIJobRequestInputsSrc()
    createAiJobReqInputSrc.data_id = data_id
    createAiJobReqInput = CreateAIJobRequestInputs()
    createAiJobReqInput.src = createAiJobReqInputSrc
    createAiJobReqInput.idx = 1
    createAiJobReq.inputs = [createAiJobReqInput]
    createAiJobReq.app = AIJobAPP.LAND_COVER_CLASSIFICATION.value
    createAiJobReq.confidence = 50
    createAiJobReq.area_threshold = 1

    createAiJobResp: CreateAIJobResponse = CLIENT.create_aijob(createAiJobReq)
    job_id = createAiJobResp.body.jobs[0].job_id

    while True:
        get_job_req: GetJobsRequest = GetJobsRequest()
        get_job_req.job_ids = [job_id]

        get_job_resp: GetJobsResponse = CLIENT.get_jobs(get_job_req)
        job_status = get_job_resp.body.list[0].status
        print(f"JobId: {job_id}, jobStatus: {try_enum(JobStatus, job_status)}")
        if job_status == JobStatus.FINISHED.value:
            out_data_id = get_job_resp.body.list[0].job_out_data_id
            out_data_type = JobOutDataType(get_job_resp.body.list[0].out_data_type)
            break
        elif job_status == JobStatus.ERROR.value:
            raise ValueError(f"{job_id} error")
        else:
            time.sleep(3)

    # 地物分类产出矢量数据
    while True:
        list_user_vector_datas_req: ListUserVectorDatasRequest = ListUserVectorDatasRequest()
        list_user_vector_datas_req.data_id = out_data_id
        list_user_vector_datas_req.from_type = UserDataFromType.RESULT.value
        list_user_vector_datas_req.page_number = 1
        list_user_vector_datas_req.page_size = 1

        list_user_vector_datas_resp: ListUserVectorDatasResponse = CLIENT.list_user_vector_datas(
            list_user_vector_datas_req)
        status = list_user_vector_datas_resp.body.list[0].vector.publish_status
        publish_msg = list_user_vector_datas_resp.body.list[0].vector.publish_msg

        print(f"outDataId: {out_data_id}, publishStatus: {status}, publishMsg: {publish_msg}")
        if status == PublishStatus.PUBLISHDONE.value:
            break
        elif status == PublishStatus.PUBLISHFAIL.value:
            raise ValueError(f"outDataId: {out_data_id} publish failed")
        else:
            time.sleep(3)

    return job_status, out_data_id


def sync_download_data(data_id: str, local_file_path: str) -> None:
    download_data: DownloadDataRequest = DownloadDataRequest()
    download_data.data_id = data_id

    while True:
        download_data_resp: DownloadDataResponse = CLIENT.download_data(download_data)
        if download_data_resp.body.finished:
            download_url = download_data_resp.body.download_url
            break
        else:
            download_status = download_data_resp.body.status
            print(f"download dataId: {data_id}, status: {download_status}")
            if download_status == DownloadStatus.FAILED:
                raise ValueError(f"download {data_id} failed")
            else:
                time.sleep(3)
    with open(local_file_path, "wb") as local_file, requests.get(download_url) as remote_stream:
        local_file.write(remote_stream.content)


if __name__ == "__main__":
    # data_id, _ = sync_pulish_raster("uploaded_by_openapi", "http://example/test.tiff")
    data_id = '8bce8aba8ac7d469e012d8a57d4c32e0'
    _, out_data_id = sync_create_lcc_job(data_id)
    sync_download_data(out_data_id, r"D:\Project\DatasetProcessor\tmp\tmp.zip")
