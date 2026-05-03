import base64

def handler(event, context):
    output = []

    for record in event['records']:
        output_record = {
            'recordId': record['recordId'],
            'result': 'Ok',
            'data': record['data']
        }
        output.append(output_record)

    return {'records': output}