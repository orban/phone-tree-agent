Discovering Voice Agent Scenarios
Your job is to test various AI Voice Agents for small businesses: home repair companies, local vet, car shop, etc.

These agents can help with simple tasks like collecting information about new customers and booking appointments.

In order to thoroughly test all the possible scenarios that the agents are capable of handling, you will need to run some discovery process that would repeatedly call each agent to discover the call tree / graph.




EXERCISE

The task for this exercise is to automate the call tree / graph discovery process.

Given a phone number of an agent and a short description about what the agent is set up to do, automate a discovery process using a series of synthetic phone call conversations.

The result should be the set of all the possible scenarios that the agent is capable of handling.

See the attached diagram as an example of a simple Voice AI Agent and the possible scenarios it can handle.






PHONE CALL ENDPOINTS

You are given a set of helper endpoints that you can use to easily start new phone calls and retrieve call recordings.

POST https://app.hamming.ai/api/rest/exercise/start-call
Authorization: Bearer <api_token>

Request Body
Response
Webhook Payload
{
  phone_number: <Number to call>
  prompt: <Agent System Prompt>
  webhook_url: <Your Webhook URL>
}
{
  id: string
}
{
  id: string
  status: string
  recording_available: bool
}



GET https://app.hamming.ai/api/media/exercise?id={id}
Response: audio/wav
DeepGram, Assembly (audio to transcript) => free trials.
AI Model is GPT-4o

Test Phone Number: +14153580761

EXPECTED OUTPUT
Either a graph structure (mermaid or textual) or a dataset with all the graph traversals is a good output

GRADING
Meets all the specified requirements in this doc
Is valid runnable code
Has appropriate usage of design patterns, data structures, concurrency
Has extendable architecture
Has console or visual outputs that allows the interviewers to follow the system progress in realtime
Has production-quality code cleanliness
Has production-quality docs on all public functions

TIMELINES & EXPECTATIONS
This take-home requires full-time, intensive effort. We generally hear back from folks within 48 hours.
You can use any programming language, framework, and IDE you’d like; however, we strongly discourage the use of microservices, cloud deployments, RPCs, DBs, etc. due to time constraints.

