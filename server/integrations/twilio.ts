import 'dotenv/config';
import twilio from 'twilio';

export async function getTwilioClient() {
  const accountSid = process.env.TWILIO_ACCOUNT_SID;
  const authToken = process.env.TWILIO_AUTH_TOKEN;

  if (!accountSid || !authToken) {
    throw new Error('Twilio credentials missing. Please set TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN in your .env file.');
  }
  
  return twilio(accountSid, authToken);
}

export async function getTwilioFromPhoneNumber() {
  const phoneNumber = process.env.TWILIO_PHONE_NUMBER;
  if (!phoneNumber) {
    throw new Error('Twilio phone number missing. Please set TWILIO_PHONE_NUMBER in your .env file.');
  }
  return phoneNumber;
}