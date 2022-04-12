import json
import os
import time

import boto3

import logging

logger = logging.getLogger(__name__)


def call_lambda(lambda_function_name, content, sync=False):
    """
    Invokes the specified lambda with the provided content
    Args:
        lambda_function_name:
        content:

    Returns:

    """
    lambdaAWS = boto3.client('lambda', region_name=os.environ.get('REGION_NAME'))

    if sync:
        content.update({"sync":True})
        response = lambdaAWS.invoke(
            FunctionName=lambda_function_name,
            InvocationType='RequestResponse',
            Payload=json.dumps(content).encode(),
            #LogType='None' | 'Tail',
            #ClientContext='string',
            #Qualifier='string'
        )
        return json.loads(response['Payload'].read().decode("utf-8"))

    else:
        lambdaAWS.invoke(
            FunctionName=lambda_function_name,
            InvocationType='Event',
            Payload=json.dumps(content).encode()
        )
        return None


class QueueProcessor():
    """
    Abstract class that will process the content of a specified queue and call the "handle_message" function on
    received messages.
    """
    def __init__(self, queue_name, region_name=None, aws_access_key_id=None, aws_secret_access_key=None):
        self.sqs = None
        try:
            self.sqs = boto3.resource('sqs',
                             region_name=region_name or os.environ.get("REGION_NAME"),
                             aws_access_key_id=aws_access_key_id or os.environ.get("AWS_ACCESS_KEY_ID"),
                             aws_secret_access_key=aws_secret_access_key or os.environ.get("AWS_SECRET_ACCESS_KEY"))

            self.queue = self.sqs.get_queue_by_name(QueueName=queue_name)
            logger.info("Queue {} loaded and ready to be pulled".format(queue_name))
        except Exception as e:
            logger.warning(str(e))
            logger.info("Failed to initialize the sqs queue. Receiving or pushing messages to queues will not be possible.")

    def receive_message(self, MaxNumberOfMessages=1):
        """
        Receiving message from a queue
        Args:
            MaxNumberOfMessages:

        Returns:

        """
        messages = self.queue.receive_messages(MaxNumberOfMessages=MaxNumberOfMessages)
        if len(messages)>0:
            logger.info("Received {} message(s)".format(len(messages)))
            for message in messages:
                body = {"success":True,
                        "error":None,
                        "data":None}
                message_body={}
                try:
                    message_body = eval(message.body)
                    document_id = message_body.get("document_id")
                    body["document_id"]=document_id
                    logger.debug("{} - Message body received and evaluated".format(document_id))
                    logger.debug(message_body)
                    message_result = self.handle_message(message_body)
                    logger.info("Message successfully processed".format(document_id))
                    logger.debug(message_result)
                    body["data"] = message_result
                except Exception as e:
                    document_id = message_body.get("document_id")
                    body["document_id"]=document_id
                    logger.warning("Error while trying to handle message on document_id {}".format(document_id))
                    logger.warning(str(e))
                    logger.exception(e)
                    body["error"] = str(e)
                    body["success"] = False
                message.delete()
                try:
                    self.send_result(body)
                except Exception as e:
                    logger.warning("Failed to forward result")
                    logger.warning("Error : {}".format(e))
                    logger.error(e)

    def send_result(self, result):
        raise NotImplementedError()

    def handle_message(self, body):
        raise NotImplementedError()


def process_queues(queues, pull_freq=1):
    """
    Process a list of provided queues, pulling every "pull_freq"
    Args:
        queues:
        pull_freq:
        log_freq:

    Returns:

    """
    logger.info("Starting queues processing for {} queue(s)".format(len(queues)))

    while True:
        for queue in queues:
            queue.receive_message()
        time.sleep(pull_freq)