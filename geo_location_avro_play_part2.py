import os
import math
import asyncio
import traceback
from helpers.log import get_logger, log_execution_time
from google.cloud import bigquery, secretmanager
from api.geo_location_api import GeoLocationAPI
from logic.utilities import Utilities
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from googlemaps import exceptions as googlemaps_exceptions
from google.api_core import exceptions as gcloud_exceptions

class GeoLocationExtractionLatLong:
    def __init__(self):
        self.logger = get_logger()
        self.bq_client = bigquery.Client()
        self.geo_location_api = GeoLocationAPI()
        self.utilities = Utilities()
        self.project_id = os.getenv("GCP_PROJECT_ID")
        self.table_id = os.getenv("GCP_BQ_TABLE_ID")
        self.dataset_id = os.getenv("GCP_BQ_DATASET_ID")
        self.query_table_id = os.getenv("GCP_BQ_QUERY_TABLE_ID")
        self.query_dataset_id = os.getenv("GCP_BQ_QUERY_DATASET_ID")
        self.intermediate_table_id = os.getenv("GCP_BQ_INTERMEDIATE_TABLE_ID")
        self.intermediate_table = f"{self.project_id}.{self.dataset_id}.{self.intermediate_table_id}"
    
    


    @log_execution_time
    def get_lat_long_from_table(self):
        try:
            query = f"""SELECT latitude, longitude, mdlzID FROM `{self.project_id}.{self.query_dataset_id}.{self.query_table_id}` WHERE pincode is null and latitude is not null and longitude is not null"""
            # query = f"""SELECT latitude, longitude,  mdlzID FROM `{self.project_id}.{self.query_dataset_id}.{self.query_table_id}`
            #             WHERE mdlzID IN (
            #             'mid_3f3986a2-96de-475a-9ec5-76057cb1642d','mid_7e10ad74-c442-4432-9e4c-a51dfda01c91','mid_592c447c-1122-49e5-8d87-e8ef7fa2dfdd','mid_2de5270c-7a7d-44ec-9479-65189e1ad427','mid_1bc9017e-4d73-40b9-8fda-7d2ba46dd9a7',
            #             'mid_139beba0-e56f-4b06-ab03-ec15b49bdfa0','mid_dbb8136f-91c4-428a-be6c-6b0679dbe0b9','mid_4c25539d-e110-485f-9689-2130a3a48ab3','mid_0a42c6ee-1f09-41ad-b220-5f83898eb0cd','mid_f967f1b2-db7d-405f-a1cd-1ca9addc1d05',
            #             'mid_6f8bb58f-6bfe-41ab-b436-45e6549cc8a6','mid_ee7d9b9f-85ec-44fc-b227-20d8134014d9','mid_0366de28-7c73-43e8-bc6c-203ff62b4011','mid_ad173f0d-e696-4637-a763-1fa716e88d23','mid_95928193-922a-4b18-b6ae-f9ff0745ce72',
            #             'mid_3d4260fc-3a4b-4ce6-9604-6a6cc63bee0c','mid_c1daf661-cf20-4a4a-9e92-469cf04a6c94','mid_58bd420f-00e1-4a16-81ed-9f68c691c63f','mid_09032563-b859-4f4a-8e1c-1a9b603db168','mid_d19e45ad-07c1-4e13-ac36-4c971e3952aa',
            #             'mid_fe87e8fe-8834-4eea-8db3-f5ae520fd4e0','mid_fc317346-a10a-40cd-8abe-338934d42f65','mid_05b3dba9-f7b0-4199-94ca-cf5aecaa9da2','mid_733896d6-b2ac-445b-8914-667ff317fc72','mid_6126f6c1-5858-43e7-972a-5b03a99742d1',
            #             'mid_c6870892-eb2c-41e2-a3dd-a436b4ce4a6a','mid_fb750c3b-5b35-4263-8db3-4ea1e7dde4ac','mid_9ea89618-aefb-4c8f-9b09-7e5b29547866','mid_0f48975c-5306-4211-baa0-2ffd5e477843','mid_303440c2-fd6c-4beb-a7a0-1985efd1b105',
            #             'mid_3950355a-da67-4256-a275-4cbfee57a77c','mid_30c0f28d-046e-4cf5-b99f-168235e10f5f','mid_d68e6c40-aa3d-4a07-8ce4-5e49ce905dc1','mid_356b4802-b348-4411-aaed-452f2663348e','mid_a8126d92-2d18-45db-8d1d-1ad62ff0d185',
            #             'mid_d26de36a-bfb5-459d-b432-e6bc22fd96c3','mid_fe2318d9-cbdc-411c-99f7-9f1f7f5cad9c','mid_0e6a2619-9365-4905-97fd-30cec9882fcc','mid_6ac5ec31-fe3d-4a9c-a4de-564f8efe985b','mid_65945aa8-2b5f-4739-ab6e-98ed66470c0f',
            #             'mid_8c21100f-9abe-45d1-a8f0-c74ab914a0fe','mid_e4560b3a-7ec4-4789-9e20-538f50720488','mid_8929121c-0acc-4936-9413-42b141956162','mid_15965dad-6a2f-4520-8c57-2df982cbfab4','mid_ac59c6c8-5cd9-4602-ac8d-92018050686c',
            #             'mid_eb5cde73-dfbc-4dd0-9afc-f1da2fd3a481','mid_dcd2e4cc-f070-4d95-a153-35c2c472b493','mid_5f6ac0ec-77b3-4f7c-8833-6f72c8a8d46b','mid_ef7029e9-04c9-40fa-8f81-ef99481c456d','mid_438f3fc5-a1e0-46ec-a51f-254c0d684a53',
            #             'mid_3fdb651b-20fc-4558-bc65-b42e7f018a89','mid_325ac6ff-24bf-45fa-887d-624034c6c0de','mid_18546e9f-7468-4bbc-882f-350f4251ef47','mid_597e0dcb-0183-4c2d-b2ac-8a1f43e4a9c9','mid_10d5719d-33c1-4877-9376-b9c0118c6a22',
            #             'mid_c7ed30dc-b2bc-4305-8e39-4caaaeeafbf8','mid_981559c6-c855-43a1-ac36-fbf821420458','mid_9455a7f1-83ca-4b7a-ad68-d2923e32463d','mid_a640b10c-7cb6-40d6-aabc-e00de0adc4ad','mid_ffa4d07f-1ef7-4cac-8c11-51f3506bf674',
            #             'mid_4fdd98ff-30ed-4543-9a18-aff93a7d91d5','mid_90caabf7-2521-40a7-9572-f7d769adcf58','mid_effb18df-294a-4555-883b-66ddf2b9e1db','mid_b5b2dd8d-60ce-410d-953c-5553011621b6','mid_8b3eb73a-68dd-4634-ac0d-1e66050905b0',
            #             'mid_c3c746ce-de73-448a-bcec-20b698abd4ba','mid_bb00b79a-293b-4464-9c6f-1e80ce65c572','mid_0ca9141d-913e-4ed1-84d2-347b69cceaf5','mid_610657d0-cbbe-43ed-9892-e71818ff8335','mid_50c1427f-167f-4833-a8f6-6a14a6cad52b',
            #             'mid_ff5ea1ed-9e7a-48b9-9599-5e3fc61c81e8','mid_915c4ec6-54e5-4cf5-959c-7dcfebb5a731','mid_bdd87ebf-6ebb-4492-af9c-f9545d039cb2','mid_fe07085d-d0b1-434a-9324-868d0aaf4231','mid_3d8070bc-629d-46ff-b7a9-ecb582b271e5',
            #             'mid_d126651a-3e56-476b-adb2-d76b79ce20cd','mid_716546a8-2400-4a97-9f86-57a93a79f82c','mid_6ed34e03-8b7a-46df-afd2-8f7b6f600784','mid_76830950-bd7a-4576-901d-2fb150e93e04','mid_69544c08-850d-4b9e-b3af-b4e214abc8b8',
            #             'mid_deea5b53-6f6b-49c1-8047-690218991ec0','mid_c6d8542c-88db-465f-b61b-e2badb7d3a1c','mid_3cfec847-a17f-4629-a745-7fa95694466f','mid_ca61cacf-739f-444e-9671-67b322b7ae91','mid_85c131a1-3176-4089-8ba6-e0583cefd1bf',
            #             'mid_2b1606b8-10fb-4a34-9067-40730e7b81b6','mid_8241ab19-8901-4963-9f1f-1956b3c12a39','mid_b073c4bb-3f98-49f3-9faf-119818565945','mid_c48e0552-5730-4ae8-8609-85f08c3a8ea2','mid_82a085bc-a7c1-4c4b-baed-4368743df86b',
            #             'mid_f805937c-f6eb-4224-85f4-767820eeb54e','mid_a8960fe9-66e1-47a4-9b37-c920d71e319f','mid_b23a3242-2497-4c27-8359-cd9e8dc43412','mid_47600215-c8dd-40a6-aafd-91ecc1d1c25c','mid_0bdbe204-287f-4753-bfe2-2c778e896471',
            #             'mid_a4950f2b-b1be-48eb-b943-653776727f19','mid_80451bc3-6617-4d55-b4a9-5e1ff6b04b92','mid_e492d988-0463-4a59-8821-5d50424ac6ba','mid_00e79b7a-fcca-4b0e-abe1-e3b722ce477a','mid_b96edf5a-c3e0-4fdd-ad08-596898e11e45')"""
            query_job = self.bq_client.query(query)
            results = query_job.result()
            lat_long_list = []
            for row in results:
                lat_long_list.append(self.utilities.normalize_row_to_dict(row))
            self.logger.info(f"Total rows returned: {results.total_rows}")
            self.logger.info(f"Sample data: {lat_long_list[:5]}")
            return lat_long_list
        except Exception as e:
            traceback.print_exc()
            self.logger.error(f"Error fetching : {e}")
            raise
    

    
    # @log_execution_time
    # def get_city_state_data_from_lat_long(self, data_list):
    #     try:
    #         state = None
    #         district = None
    #         city = None
    #         latitude = None
    #         longitude = None
    #         lat_long_dict = {}
    #         enriched_data = []
    #         not_enriched_data = []
    #         for data in data_list:
    #             if isinstance(data, dict):
    #                 latitude = data.get("latitude")
    #                 longitude = data.get("longitude")
    #                 mdlzID = data.get("mdlzID")
    #             geolocation_data = self.geo_location_api.get_geolocation_data_from_lat_long(latitude, longitude)
    #             address_components = geolocation_data[0]['address_components']
    #             if geolocation_data:
    #                 for component in address_components:
    #                     if 'administrative_area_level_1' in component['types']:
    #                         state = component['long_name']
    #                     if 'administrative_area_level_2' in component['types']:
    #                         district = component['long_name']
    #                     if 'administrative_area_level_3' in component['types']:
    #                         city = component['long_name']
    #                 lat_long_dict =  {
    #                     "mdlzID": mdlzID,
    #                     "state": state,
    #                     "district": district,
    #                     "city": city,
    #                     "latitude": latitude,
    #                     "longitude": longitude
    #                 }
    #                 enriched_data.append(lat_long_dict)
    #                 self.logger.info(f"Enriched data: {len(enriched_data)}")
    #             else:
    #                 lat_long_dict =  {
    #                         "latitude": latitude,
    #                         "longitude": longitude,
    #                         "mdlzID": mdlzID,
    #                         "state": None,
    #                         "district": None,
    #                         "city": None
    #                     }
    #                 not_enriched_data.append(mdlzID)
    #                 self.logger.info(f"Not enriched data: {not_enriched_data}")
    #         return enriched_data, not_enriched_data
    #     except Exception as e:
    #         traceback.print_exc()
    #         self.logger.error(f"Error fetching : {e}")
    #         raise
    
    

    @log_execution_time
    def get_city_state_data_from_lat_long(self, data_list):
        try:
            state = None
            district = None
            city = None
            latitude = None
            longitude = None
            lat_long_dict = {}
            enriched_data = []
            not_enriched_data = []
            for data in data_list:
                # reset per-row variables (minimal change)
                state = None
                district = None
                city = None

                if isinstance(data, dict):
                    latitude = data.get("latitude")
                    longitude = data.get("longitude")
                    mdlzID = data.get("mdlzID")

                # call geo API safely per-row
                try:
                    geolocation_data = self.geo_location_api.get_geolocation_data_from_lat_long(latitude, longitude)
                except googlemaps_exceptions.HTTPError as he:
                    # catch HTTP errors from Google Maps and treat this row as not enriched
                    self.logger.warning(f"GoogleMaps HTTPError for mdlzID={mdlzID} coords=({latitude},{longitude}): {he}. Marking nulls and continuing.")
                    lat_long_dict = {
                        "latitude": latitude,
                        "longitude": longitude,
                        "mdlzID": mdlzID,
                        "state": None,
                        "district": None,
                        "city": None
                    }
                    enriched_data.append(lat_long_dict)
                    not_enriched_data.append(mdlzID)
                    continue
                except Exception as e:
                    # any other per-row failure: log and continue with nulls
                    self.logger.exception(f"Error fetching geolocation for mdlzID={mdlzID} coords=({latitude},{longitude}): {e}. Marking nulls and continuing.")
                    lat_long_dict = {
                        "latitude": latitude,
                        "longitude": longitude,
                        "mdlzID": mdlzID,
                        "state": None,
                        "district": None,
                        "city": None
                    }
                    enriched_data.append(lat_long_dict)
                    not_enriched_data.append(mdlzID)
                    continue

                # if API returned nothing, mark as not enriched
                if not geolocation_data:
                    lat_long_dict = {
                        "latitude": latitude,
                        "longitude": longitude,
                        "mdlzID": mdlzID,
                        "state": None,
                        "district": None,
                        "city": None
                    }
                    enriched_data.append(lat_long_dict)
                    not_enriched_data.append(mdlzID)
                    continue

                # Safely access address_components (guard against odd shapes)
                try:
                    address_components = geolocation_data[0].get('address_components', [])
                except Exception:
                    address_components = []

                # extract values from components (single pass)
                for component in address_components:
                    if 'administrative_area_level_1' in component.get('types', []):
                        state = component.get('long_name')
                    if 'administrative_area_level_2' in component.get('types', []):
                        district = component.get('long_name')
                    if 'administrative_area_level_3' in component.get('types', []):
                        city = component.get('long_name')

                lat_long_dict = {
                    "mdlzID": mdlzID,
                    "state": state,
                    "district": district,
                    "city": city,
                    "latitude": latitude,
                    "longitude": longitude
                }
                enriched_data.append(lat_long_dict)
                self.logger.info(f"Enriched data: {len(enriched_data)}")
            return enriched_data, not_enriched_data
        except Exception as e:
            traceback.print_exc()
            self.logger.error(f"Error fetching : {e}")
            raise

    

    # def main(self):
    #     try:
    #         size_of_chunk = 200
    #         self.logger.info(f"Geo location job started successfully")
    #         data_list = self.get_lat_long_from_table()
    #         self.logger.info(f"Total data_list: {len(data_list)}")
    #         total_chunks = math.ceil(len(data_list) / size_of_chunk)
    #         self.logger.info(f"Total chunks: {total_chunks}")
    #         if data_list:
    #             for chunk_id, chunk in self.utilities.create_chunks(data_list, size_of_chunk):
    #                 self.logger.info(f"Doing for Chunk id: {chunk_id}/{total_chunks}) with each chunk size: {len(chunk)}")
    #                 enriched_data, not_enriched_data = self.get_city_state_data_from_lat_long(chunk)
    #                 self.logger.info(f"Total enriched data: {len(enriched_data)}")
    #                 self.logger.info(f"enriched data: {enriched_data}")
    #                 self.logger.info(f"Total not enriched data: {len(not_enriched_data)}")
    #                 self.logger.info(f"Total not enriched data: {not_enriched_data}")
    #                 if enriched_data:
    #                     file_path = self.utilities.write_fastavro_file(chunk_id, enriched_data)
    #                     # self.utilities.upload_to_gcs(chunk_id, file_path)
    #                     avro_schema = self.utilities.create_avro_schema()
    #                     self.logger.info(f"avro_schema: {avro_schema}")
    #                     temp_table_id = self.utilities.create_temp_table_from_avro(avro_schema)
    #                     # self.utilities.upload_avro_file_from_gcs_to_bigquery_table(file_path, temp_table_id)
    #                     self.utilities.upload_avro_file_from_local_to_bigquery(file_path, temp_table_id)
    #                     self.logger.info(f"Data written to BigQuery successfully")
    #                     self.utilities.update_harmonized_table_query(temp_table_id)
    #                     self.logger.info(f"Data written to Harmonized table: {self.project_id}.{self.query_dataset_id}.{self.query_table_id} successfully")
    #                     os.remove(file_path)
    #                     self.logger.info(f"File {file_path} deleted successfully")
    #                     self.logger.info(f"Chunk {chunk_id} completed successfully")
    #                     self.utilities.delete_temp_table(temp_table_id)
    #                     self.logger.info(f"Temp table {temp_table_id} deleted successfully")
    #                     enriched_data = []
        
    #     except Exception:
    #         traceback.print_exc()
    #         self.logger.error("Error fetching data from BigQuery")
    #         raise




    # def main(self):
    #     try:
    #         size_of_chunk = 20
    #         max_workers = 5  # number of concurrent chunk workers
    #         self.logger.info("Geo location job started successfully")

    #         data_list = self.get_lat_long_from_table()
    #         total_rows = len(data_list)
    #         self.logger.info(f"Total data_list: {total_rows}")

    #         if not data_list:
    #             self.logger.info("No rows to process. Exiting.")
    #             return

    #         # Build list of chunks (so we can log total_chunks and reference chunk index)
    #         chunks = list(self.utilities.create_chunks(data_list, size_of_chunk))
    #         total_chunks = len(chunks)
    #         self.logger.info(f"Total chunks: {total_chunks} (chunk size up to {size_of_chunk})")

    #         # Worker function: processes a single chunk end-to-end
    #         def process_chunk(chunk_id, chunk, total_chunks_local):
    #             thread_name = threading.current_thread().name
    #             start_ts = time.time()
    #             start_human = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_ts))
    #             self.logger.info(f"[{thread_name}] Starting chunk {chunk_id}/{total_chunks_local} at {start_human} (rows={len(chunk)})")

    #             file_path = None
    #             temp_table_id = None
    #             try:
    #                 # 1) Enrich
    #                 enriched_data, not_enriched_data = self.get_city_state_data_from_lat_long(chunk)
    #                 self.logger.info(f"[{thread_name}] Chunk {chunk_id}: enriched={len(enriched_data)}, not_enriched={len(not_enriched_data)}")

    #                 if not enriched_data:
    #                     self.logger.info(f"[{thread_name}] Chunk {chunk_id}: nothing to write, finishing.")
    #                     return {"chunk_id": chunk_id, "status": "no_data", "rows": len(chunk)}

    #                 # 2) Write avro using fastavro (your existing util)
    #                 file_path = self.utilities.write_fastavro_file(chunk_id, enriched_data)
    #                 self.logger.info(f"[{thread_name}] Chunk {chunk_id}: avro written: {file_path}")

    #                 # 3) Create temp table from avro schema
    #                 avro_schema = self.utilities.create_avro_schema()
    #                 self.logger.info(f"[{thread_name}] Chunk {chunk_id}: avro_schema: {avro_schema}")
    #                 temp_table_id = self.utilities.create_temp_table_from_avro(avro_schema)
    #                 self.logger.info(f"[{thread_name}] Chunk {chunk_id}: temp table created: {temp_table_id}")

    #                 # 4) Upload avro to BigQuery
    #                 self.utilities.upload_avro_file_from_local_to_bigquery(file_path, temp_table_id)
    #                 self.logger.info(f"[{thread_name}] Chunk {chunk_id}: uploaded avro to {temp_table_id}")

    #                 # 5) Update harmonized table via provided util
    #                 self.utilities.update_harmonized_table_query(temp_table_id)
    #                 self.logger.info(f"[{thread_name}] Chunk {chunk_id}: harmonized table updated from {temp_table_id}")

    #                 end_ts = time.time()
    #                 duration = end_ts - start_ts
    #                 self.logger.info(f"[{thread_name}] Completed chunk {chunk_id}/{total_chunks_local} in {duration:.1f}s")
    #                 return {
    #                     "chunk_id": chunk_id,
    #                     "status": "success",
    #                     "rows": len(chunk),
    #                     "enriched_rows": len(enriched_data),
    #                     "not_enriched_rows": len(not_enriched_data),
    #                     "duration_seconds": duration,
    #                 }

    #             except Exception as e:
    #                 # Log the exception and return failure info for this chunk
    #                 traceback_str = traceback.format_exc()
    #                 self.logger.error(f"[{thread_name}] Chunk {chunk_id} failed: {e}\n{traceback_str}")
    #                 return {"chunk_id": chunk_id, "status": "failed", "error": str(e)}

    #             finally:
    #                 # Best-effort cleanup: delete avro file and delete temp table
    #                 try:
    #                     if file_path and os.path.exists(file_path):
    #                         try:
    #                             os.remove(file_path)
    #                             self.logger.info(f"[{thread_name}] Chunk {chunk_id}: deleted local file {file_path}")
    #                         except Exception as e_rm:
    #                             self.logger.warning(f"[{thread_name}] Chunk {chunk_id}: failed to delete file {file_path}: {e_rm}")
    #                 except Exception:
    #                     pass

    #                 try:
    #                     if temp_table_id:
    #                         try:
    #                             self.utilities.delete_temp_table(temp_table_id)
    #                             self.logger.info(f"[{thread_name}] Chunk {chunk_id}: deleted temp table {temp_table_id}")
    #                         except Exception as e_del:
    #                             self.logger.warning(f"[{thread_name}] Chunk {chunk_id}: failed to delete temp table {temp_table_id}: {e_del}")
    #                 except Exception:
    #                     pass

    #         # Submit all chunk tasks to executor; executor will run up to max_workers at a time.
    #         futures = []
    #         with ThreadPoolExecutor(max_workers=max_workers) as executor:
    #             for (chunk_id, chunk) in chunks:
    #                 # Submit process_chunk for each chunk
    #                 fut = executor.submit(process_chunk, chunk_id, chunk, total_chunks)
    #                 futures.append(fut)

    #             # As tasks complete, log their results
    #             for fut in as_completed(futures):
    #                 try:
    #                     result = fut.result()
    #                     # result is the dict returned by process_chunk
    #                     if result is None:
    #                         self.logger.warning("Received empty result for a chunk task.")
    #                         continue

    #                     status = result.get("status")
    #                     chunk_id_res = result.get("chunk_id")
    #                     if status == "success":
    #                         self.logger.info(f"Chunk {chunk_id_res} finished successfully: {result}")
    #                     elif status == "no_data":
    #                         self.logger.info(f"Chunk {chunk_id_res} had no enriched rows - skipped.")
    #                     elif status == "failed":
    #                         self.logger.error(f"Chunk {chunk_id_res} failed: {result.get('error')}")
    #                     else:
    #                         self.logger.info(f"Chunk {chunk_id_res} result: {result}")

    #                 except Exception as e:
    #                     # This catches exceptions raised when calling fut.result()
    #                     self.logger.error(f"Chunk future raised an exception: {e}\n{traceback.format_exc()}")

    #         self.logger.info("All chunks processed.")

    #     except Exception:
    #         traceback.print_exc()
    #         self.logger.error("Error in main geo location job")
    #         raise

    

    def main(self):
        try:
            size_of_chunk = 20000
            max_workers = 5
            BQ_UPDATE_LOCK = threading.Lock()
            GCS_UPLOAD_LOCK = threading.Lock()
            GCS_TO_BQ_LOCK = threading.Lock()
            self.logger.info("Geo location job started successfully")

            data_list = self.get_lat_long_from_table()
            total_rows = len(data_list)
            self.logger.info(f"Total data_list: {total_rows}")

            if not data_list:
                self.logger.info("No rows to process. Exiting.")
                return

            # Build list of chunks (so we can log total_chunks and reference chunk index)
            chunks = list(self.utilities.create_chunks(data_list, size_of_chunk))
            total_chunks = len(chunks)
            self.logger.info(f"Total chunks: {total_chunks} (chunk size up to {size_of_chunk})")

            # Worker function: processes a single chunk end-to-end
            def process_chunk(chunk_id, chunk, total_chunks_local):
                thread_name = threading.current_thread().name
                start_ts = time.time()
                start_human = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_ts))
                self.logger.info(f"[{thread_name}] Starting chunk {chunk_id}/{total_chunks_local} at {start_human} (rows={len(chunk)})")

                file_path = None
                temp_table_id = None
                try:
                    # 1) Enrich
                    enriched_data, not_enriched_data = self.get_city_state_data_from_lat_long(chunk)
                    self.logger.info(f"[{thread_name}] Chunk {chunk_id}: enriched={len(enriched_data)}, not_enriched={len(not_enriched_data)}")

                    if not enriched_data:
                        self.logger.info(f"[{thread_name}] Chunk {chunk_id}: nothing to write, finishing.")
                        return {"chunk_id": chunk_id, "status": "no_data", "rows": len(chunk)}

                    # 2) Write avro using fastavro (your existing util)
                    file_path = self.utilities.write_fastavro_file(chunk_id, enriched_data)
                    self.logger.info(f"[{thread_name}] Chunk {chunk_id}: avro written: {file_path}")

                    # 3) Create temp table from avro schema
                    # avro_schema = self.utilities.create_avro_schema()
                    # self.logger.info(f"[{thread_name}] Chunk {chunk_id}: avro_schema: {avro_schema}")
                    # temp_table_id = self.utilities.create_temp_table_from_avro(avro_schema)
                    intermediate_table = f"{self.project_id}.{self.dataset_id}.{self.intermediate_table_id}"
                    self.logger.info(f"[{thread_name}] Chunk {chunk_id}: intermediate table: {intermediate_table}")
                    # self.logger.info(f"[{thread_name}] Chunk {chunk_id}: temp table created: {temp_table_id}")

                    #  Upload avro to GCS
                    with GCS_UPLOAD_LOCK:
                        blob_name = self.utilities.upload_to_gcs(chunk_id, file_path)
                        time.sleep(2)
                    self.logger.info(f"[{thread_name}] Chunk {chunk_id}: uploaded avro file {file_path} to GCS")
                    with GCS_TO_BQ_LOCK:
                        self.utilities.upload_avro_file_from_gcs_to_bigquery_table(blob_name, intermediate_table)
                        time.sleep(2)
                    self.logger.info(f"[{thread_name}] Chunk {chunk_id}: uploaded avro to {intermediate_table}")
                    # 4) Upload avro to BigQuery
                    # self.utilities.upload_avro_file_from_local_to_bigquery(file_path, temp_table_id)
                    # self.utilities.upload_avro_file_from_local_to_bigquery(file_path, intermediate_table)
                    # self.logger.info(f"[{thread_name}] Chunk {chunk_id}: uploaded avro to {temp_table_id}")
                    # self.logger.info(f"[{thread_name}] Chunk {chunk_id}: uploaded avro to {intermediate_table}")

                    # 5) Update harmonized table via provided util
                    # ----- RETRY LOGIC FOR CONCURRENT UPDATE ERROR -----
                    with BQ_UPDATE_LOCK:
                        self.utilities.update_harmonized_table_query(temp_table_id)
                        time.sleep(5)
                    # max_update_attempts = 5
                    # base_backoff_seconds = 2
                    # update_succeeded = False

                    # for attempt in range(1, max_update_attempts + 1):
                    #     try:
                    #         self.utilities.update_harmonized_table_query(temp_table_id)
                    #         update_succeeded = True
                    #         self.logger.info(f"[{thread_name}] Chunk {chunk_id}: harmonized table updated from {temp_table_id} on attempt {attempt}")
                    #         break
                    #     except gcloud_exceptions.BadRequest as be:
                    #         msg = str(be)
                    #         # detect the BigQuery serialization / concurrent update message
                    #         if "Could not serialize access to table" in msg or "concurrent update" in msg:
                    #             sleep_time = base_backoff_seconds * (2 ** (attempt - 1))
                    #             self.logger.warning(
                    #                 f"[{thread_name}] Chunk {chunk_id}: serialization/concurrency error when updating harmonized table on attempt {attempt}. "
                    #                 f"Sleeping {sleep_time}s then retrying..."
                    #             )
                    #             time.sleep(sleep_time)
                    #             continue
                    #         else:
                    #             # not the serialization error: re-raise so it's handled by outer except
                    #             raise
                    #     except Exception:
                    #         # other unexpected error — re-raise to be handled by outer except
                    #         raise

                    # if not update_succeeded:
                    #     # All retries failed due to serialization/concurrency; raise an error so chunk is logged as failed.
                    #     raise RuntimeError(f"Chunk {chunk_id}: failed to update harmonized table {temp_table_id} after {max_update_attempts} attempts due to concurrent updates.")

                    self.logger.info(f"[{thread_name}] Chunk {chunk_id}: harmonized table updated from {temp_table_id}")

                    end_ts = time.time()
                    duration = end_ts - start_ts
                    self.logger.info(f"[{thread_name}] Completed chunk {chunk_id}/{total_chunks_local} in {duration:.1f}s")
                    return {
                        "chunk_id": chunk_id,
                        "status": "success",
                        "rows": len(chunk),
                        "enriched_rows": len(enriched_data),
                        "not_enriched_rows": len(not_enriched_data),
                        "duration_seconds": duration,
                    }

                except Exception as e:
                    # Log the exception and return failure info for this chunk
                    traceback_str = traceback.format_exc()
                    self.logger.error(f"[{thread_name}] Chunk {chunk_id} failed: {e}\n{traceback_str}")
                    return {"chunk_id": chunk_id, "status": "failed", "error": str(e)}

                finally:
                    # Best-effort cleanup: delete avro file and delete temp table
                    try:
                        if file_path and os.path.exists(file_path):
                            try:
                                os.remove(file_path)
                                self.logger.info(f"[{thread_name}] Chunk {chunk_id}: deleted local file {file_path}")
                            except Exception as e_rm:
                                self.logger.warning(f"[{thread_name}] Chunk {chunk_id}: failed to delete file {file_path}: {e_rm}")
                    except Exception:
                        pass

                    try:
                        if temp_table_id:
                            try:
                                self.utilities.delete_temp_table(temp_table_id)
                                self.logger.info(f"[{thread_name}] Chunk {chunk_id}: deleted temp table {temp_table_id}")
                            except Exception as e_del:
                                self.logger.warning(f"[{thread_name}] Chunk {chunk_id}: failed to delete temp table {temp_table_id}: {e_del}")
                    except Exception:
                        pass

            # Submit all chunk tasks to executor; executor will run up to max_workers at a time.
            futures = []
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                for (chunk_id, chunk) in chunks:
                    # Submit process_chunk for each chunk
                    fut = executor.submit(process_chunk, chunk_id, chunk, total_chunks)
                    futures.append(fut)

                # As tasks complete, log their results
                for fut in as_completed(futures):
                    try:
                        result = fut.result()
                        # result is the dict returned by process_chunk
                        if result is None:
                            self.logger.warning("Received empty result for a chunk task.")
                            continue

                        status = result.get("status")
                        chunk_id_res = result.get("chunk_id")
                        if status == "success":
                            self.logger.info(f"Chunk {chunk_id_res} finished successfully: {result}")
                        elif status == "no_data":
                            self.logger.info(f"Chunk {chunk_id_res} had no enriched rows - skipped.")
                        elif status == "failed":
                            self.logger.error(f"Chunk {chunk_id_res} failed: {result.get('error')}")
                        else:
                            self.logger.info(f"Chunk {chunk_id_res} result: {result}")

                    except Exception as e:
                        # This catches exceptions raised when calling fut.result()
                        self.logger.error(f"Chunk future raised an exception: {e}\n{traceback.format_exc()}")

            self.logger.info("All chunks processed.")
            self.logger.info("Final bigquery query to update the table harmonized table")
            self.utilities.update_harmonized_table_query(self.intermediate_table)

        except Exception:
            traceback.print_exc()
            self.logger.error("Error in main geo location job")
            raise
