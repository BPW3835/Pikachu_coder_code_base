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
    
    


    @log_execution_time
    def get_lat_long_from_table(self):
        try:
            query = f"""SELECT latitude, longitude, mdlzID FROM `{self.project_id}.{self.query_dataset_id}.{self.query_table_id}` WHERE pincode is null and latitude is not null and longitude is not null LIMIT 1000"""
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
    #         size_of_chunk = 2000
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
            size_of_chunk = 200
            max_workers = 5
            BQ_UPDATE_LOCK = threading.Lock()
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
                    avro_schema = self.utilities.create_avro_schema()
                    self.logger.info(f"[{thread_name}] Chunk {chunk_id}: avro_schema: {avro_schema}")
                    temp_table_id = self.utilities.create_temp_table_from_avro(avro_schema)
                    self.logger.info(f"[{thread_name}] Chunk {chunk_id}: temp table created: {temp_table_id}")

                    # 4) Upload avro to BigQuery
                    self.utilities.upload_avro_file_from_local_to_bigquery(file_path, temp_table_id)
                    self.logger.info(f"[{thread_name}] Chunk {chunk_id}: uploaded avro to {temp_table_id}")

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

        except Exception:
            traceback.print_exc()
            self.logger.error("Error in main geo location job")
            raise
