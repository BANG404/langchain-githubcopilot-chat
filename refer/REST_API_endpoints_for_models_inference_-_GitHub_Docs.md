# REST API endpoints for models inference - GitHub Docs

> 来源: https://docs.github.com/en/rest/models/inference?apiVersion=2026-03-10
> 摘要: Use the REST API to submit a chat completion request to a specified model, with or without organizational attribution.
> 站点: GitHub Docs

---

[Skip to main content](#main-content)

This article is also available in [Simplified Chinese](/zh/rest/models/inference).

[GitHub Docs](/en)

Version: Free, Pro, & Team

Search or ask Copilot

Search or askCopilot

Select language: current language is English

Search or ask Copilot

Search or askCopilot

Open menu

Open Sidebar

*   [REST API](/en/rest "REST API")/
*   [Models](/en/rest/models "Models")/
*   [Inference](/en/rest/models/inference "Inference")

[Home](/en)

## [REST API](/en/rest)

API Version: 2026-03-10 (latest)

*   [Quickstart](/en/rest/quickstart)
*   About the REST API
    *   [About the REST API](/en/rest/about-the-rest-api/about-the-rest-api)
    *   [Comparing GitHub's APIs](/en/rest/about-the-rest-api/comparing-githubs-rest-api-and-graphql-api)
    *   [API Versions](/en/rest/about-the-rest-api/api-versions)
    *   [Breaking changes](/en/rest/about-the-rest-api/breaking-changes)
    *   [OpenAPI description](/en/rest/about-the-rest-api/about-the-openapi-description-for-the-rest-api)
*   Using the REST API
    *   [Getting started](/en/rest/using-the-rest-api/getting-started-with-the-rest-api)
    *   [Rate limits](/en/rest/using-the-rest-api/rate-limits-for-the-rest-api)
    *   [Pagination](/en/rest/using-the-rest-api/using-pagination-in-the-rest-api)
    *   [Libraries](/en/rest/using-the-rest-api/libraries-for-the-rest-api)
    *   [Best practices](/en/rest/using-the-rest-api/best-practices-for-using-the-rest-api)
    *   [Troubleshooting](/en/rest/using-the-rest-api/troubleshooting-the-rest-api)
    *   [Timezones](/en/rest/using-the-rest-api/timezones-and-the-rest-api)
    *   [CORS and JSONP](/en/rest/using-the-rest-api/using-cors-and-jsonp-to-make-cross-origin-requests)
    *   [Issue event types](/en/rest/using-the-rest-api/issue-event-types)
    *   [GitHub event types](/en/rest/using-the-rest-api/github-event-types)
*   Authentication
    *   [Authenticating](/en/rest/authentication/authenticating-to-the-rest-api)
    *   [Keeping API credentials secure](/en/rest/authentication/keeping-your-api-credentials-secure)
    *   [Endpoints for GitHub App installation tokens](/en/rest/authentication/endpoints-available-for-github-app-installation-access-tokens)
    *   [Endpoints for GitHub App user tokens](/en/rest/authentication/endpoints-available-for-github-app-user-access-tokens)
    *   [Endpoints for fine-grained PATs](/en/rest/authentication/endpoints-available-for-fine-grained-personal-access-tokens)
    *   [Permissions for GitHub Apps](/en/rest/authentication/permissions-required-for-github-apps)
    *   [Permissions for fine-grained PATs](/en/rest/authentication/permissions-required-for-fine-grained-personal-access-tokens)
*   Guides
    *   [Script with JavaScript](/en/rest/guides/scripting-with-the-rest-api-and-javascript)
    *   [Script with Ruby](/en/rest/guides/scripting-with-the-rest-api-and-ruby)
    *   [Discover resources for a user](/en/rest/guides/discovering-resources-for-a-user)
    *   [Delivering deployments](/en/rest/guides/delivering-deployments)
    *   [Rendering data as graphs](/en/rest/guides/rendering-data-as-graphs)
    *   [Working with comments](/en/rest/guides/working-with-comments)
    *   [Building a CI server](/en/rest/guides/building-a-ci-server)
    *   [Get started - Git database](/en/rest/guides/using-the-rest-api-to-interact-with-your-git-database)
    *   [Get started - Checks](/en/rest/guides/using-the-rest-api-to-interact-with-checks)
    *   [Encrypt secrets](/en/rest/guides/encrypting-secrets-for-the-rest-api)

---

*   Actions
    *   Artifacts
    *   Cache
    *   GitHub-hosted runners
    *   OIDC
    *   Permissions
    *   Secrets
    *   Self-hosted runner groups
    *   Self-hosted runners
    *   Variables
    *   Workflow jobs
    *   Workflow runs
    *   Workflows
*   Activity
    *   Events
    *   Feeds
    *   Notifications
    *   Starring
    *   Watching
*   Apps
    *   GitHub Apps
    *   Installations
    *   Marketplace
    *   OAuth authorizations
    *   Webhooks
*   Billing
    *   Budgets
    *   Billing usage
*   Branches
    *   Branches
    *   Protected branches
*   Campaigns
    *   Security campaigns
*   Checks
    *   Check runs
    *   Check suites
*   Classroom
    *   Classroom
*   Code scanning
    *   Code scanning
*   Code security settings
    *   Configurations
*   Codes of conduct
    *   Codes of conduct
*   Codespaces
    *   Codespaces
    *   Organizations
    *   Organization secrets
    *   Machines
    *   Repository secrets
    *   User secrets
*   Collaborators
    *   Collaborators
    *   Invitations
*   Commits
    *   Commits
    *   Commit comments
    *   Commit statuses
*   Copilot
    *   Copilot content exclusion management
    *   Copilot metrics
    *   Copilot user management
*   Credentials
    *   Revocation
*   Dependabot
    *   Alerts
    *   Repository access
    *   Secrets
*   Dependency graph
    *   Dependency review
    *   Dependency submission
    *   Software bill of materials (SBOM)
*   Deploy keys
    *   Deploy keys
*   Deployments
    *   Deployment branch policies
    *   Deployments
    *   Environments
    *   Protection rules
    *   Deployment statuses
*   Emojis
    *   Emojis
*   Enterprise teams
    *   Enterprise team members
    *   Enterprise team organizations
    *   Enterprise teams
*   Gists
    *   Gists
    *   Comments
*   Git database
    *   Blobs
    *   Commits
    *   References
    *   Tags
    *   Trees
*   Gitignore
    *   Gitignore
*   Interactions
    *   Organization
    *   Repository
    *   User
*   Issues
    *   Assignees
    *   Comments
    *   Events
    *   Issue dependencies
    *   Issue field values
    *   Issues
    *   Labels
    *   Milestones
    *   Sub-issues
    *   Timeline
*   Licenses
    *   Licenses
*   Markdown
    *   Markdown
*   Meta
    *   Meta
*   Metrics
    *   Community
    *   Statistics
    *   Traffic
*   Migrations
    *   Organizations
    *   Source endpoints
    *   Users
*   Models
    *   Catalog
    *   Embeddings
    *   Inference
        *   [About GitHub Models inference](#about-github-models-inference)
        *   [Run an inference request attributed to an organization](#run-an-inference-request-attributed-to-an-organization)
        *   [Run an inference request](#run-an-inference-request)
*   Organizations
    *   API Insights
    *   Artifact metadata
    *   Artifact attestations
    *   Blocking users
    *   Custom properties
    *   Issue fields
    *   Issue types
    *   Members
    *   Network configurations
    *   Organization roles
    *   Organizations
    *   Outside collaborators
    *   Personal access tokens
    *   Rule suites
    *   Rules
    *   Security managers
    *   Webhooks
*   Packages
    *   Packages
*   Pages
    *   Pages
*   Private registries
    *   Organization configurations
*   Projects
    *   Draft Project items
    *   Project fields
    *   Project items
    *   Projects
    *   Project views
*   Pull requests
    *   Pull requests
    *   Review comments
    *   Review requests
    *   Reviews
*   Rate limit
    *   Rate limit
*   Reactions
    *   Reactions
*   Releases
    *   Releases
    *   Release assets
*   Repositories
    *   Attestations
    *   Autolinks
    *   Contents
    *   Custom properties
    *   Forks
    *   Repositories
    *   Rule suites
    *   Rules
    *   Webhooks
*   Search
    *   Search
*   Secret scanning
    *   Push protection
    *   Secret scanning
*   Security advisories
    *   Global security advisories
    *   Repository security advisories
*   Teams
    *   Members
    *   Teams
*   Users
    *   Attestations
    *   Blocking users
    *   Emails
    *   Followers
    *   GPG keys
    *   Git SSH keys
    *   Social accounts
    *   SSH signing keys
    *   Users

The REST API is now versioned. For more information, see "[About API versioning](/rest/overview/api-versions)."

*   [REST API](/en/rest "REST API")/
*   [Models](/en/rest/models "Models")/
*   [Inference](/en/rest/models/inference "Inference")

# REST API endpoints for models inference

Use the REST API to submit a chat completion request to a specified model, with or without organizational attribution.

## [About GitHub Models inference](#about-github-models-inference)

You can use the REST API to run inference requests using the GitHub Models platform. The API requires the `models: read` scope when using a fine-grained personal access token or when authenticating using a GitHub App.

The API supports:

*   Accessing top models from OpenAI, DeepSeek, Microsoft, Llama, and more.
*   Running chat-based inference requests with full control over sampling and response parameters.
*   Streaming or non-streaming completions.
*   Organizational attribution and usage tracking.

## [Run an inference request attributed to an organization](#run-an-inference-request-attributed-to-an-organization)

This endpoint allows you to run an inference request attributed to a specific organization. You must be a member of the organization and have enabled models to use this endpoint. The token used to authenticate must have the `models: read` permission if using a fine-grained PAT or GitHub App minted token. The request body should contain the model ID and the messages for the chat completion request. The response will include either a non-streaming or streaming response based on the request parameters.

### [Parameters for "Run an inference request attributed to an organization"](#run-an-inference-request-attributed-to-an-organization--parameters)

Headers

Name, Type, Description

`content-type` string Required

Setting to `application/json` is required.

`accept` string

Setting to `application/vnd.github+json` is recommended.

Path parameters

Name, Type, Description

`org` string Required

The organization login associated with the organization to which the request is to be attributed.

Query parameters

Name, Type, Description

`api-version` string

The API version to use. Optional, but required for some features.

Body parameters

Name, Type, Description

`model` string Required

ID of the specific model to use for the request. The model ID should be in the format of {publisher}/{model\_name} where "openai/gpt-4.1" is an example of a model ID. You can find supported models in the catalog/models endpoint.

`messages` array of objects Required

The collection of context messages associated with this chat completion request. Typical usage begins with a chat message for the System role that provides instructions for the behavior of the assistant, followed by alternating messages between the User and Assistant roles.

Properties of `messages`

Name, Type, Description

`role` string Required

The chat role associated with this message

Can be one of: `assistant`, `developer`, `system`, `user`

`content` string Required

The content of the message

`frequency_penalty` number

A value that influences the probability of generated tokens appearing based on their cumulative frequency in generated text. Positive values will make tokens less likely to appear as their frequency increases and decrease the likelihood of the model repeating the same statements verbatim. Supported range is \[-2, 2\].

`max_tokens` integer

The maximum number of tokens to generate in the completion. The token count of your prompt plus max\_tokens cannot exceed the model's context length. For example, if your prompt is 100 tokens and you set max\_tokens to 50, the API will return a completion with a maximum of 50 tokens.

`modalities` array of strings

The modalities that the model is allowed to use for the chat completions response. The default modality is text. Indicating an unsupported modality combination results in a 422 error. Supported values are: `text`, `audio`

`presence_penalty` number

A value that influences the probability of generated tokens appearing based on their existing presence in generated text. Positive values will make tokens less likely to appear when they already exist and increase the model's likelihood to output new tokens. Supported range is \[-2, 2\].

`response_format` object

The desired format for the response.

Can be one of these objects:

Name, Type, Description

`Object` object

Properties of `Object`

Name, Type, Description

`type` string

Can be one of: `text`, `json_object`

`Schema for structured JSON response` object

Properties of `Schema for structured JSON response`

Name, Type, Description

`type` string Required

The type of the response.

Value: `json_schema`

`json_schema` object Required

The JSON schema for the response.

`seed` integer

If specified, the system will make a best effort to sample deterministically such that repeated requests with the same seed and parameters should return the same result. Determinism is not guaranteed.

`stream` boolean

A value indicating whether chat completions should be streamed for this request.

Default: `false`

`stream_options` object

Whether to include usage information in the response. Requires stream to be set to true.

Properties of `stream_options`

Name, Type, Description

`include_usage` boolean

Whether to include usage information in the response.

Default: `false`

`stop` array of strings

A collection of textual sequences that will end completion generation.

`temperature` number

The sampling temperature to use that controls the apparent creativity of generated completions. Higher values will make output more random while lower values will make results more focused and deterministic. It is not recommended to modify temperature and top\_p for the same completion request as the interaction of these two settings is difficult to predict. Supported range is \[0, 1\]. Decimal values are supported.

`tool_choice` string

If specified, the model will configure which of the provided tools it can use for the chat completions response.

Can be one of: `auto`, `required`, `none`

`tools` array of objects

A list of tools the model may request to call. Currently, only functions are supported as a tool. The model may respond with a function call request and provide the input arguments in JSON format for that function.

Properties of `tools`

Name, Type, Description

`function` object

Properties of `function`

Name, Type, Description

`name` string

The name of the function to be called.

`description` string

A description of what the function does. The model will use this description when selecting the function and interpreting its parameters.

`parameters`

The parameters the function accepts, described as a JSON Schema object.

`type` string

Value: `function`

`top_p` number

An alternative to sampling with temperature called nucleus sampling. This value causes the model to consider the results of tokens with the provided probability mass. As an example, a value of 0.15 will cause only the tokens comprising the top 15% of probability mass to be considered. It is not recommended to modify temperature and top\_p for the same request as the interaction of these two settings is difficult to predict. Supported range is \[0, 1\]. Decimal values are supported.

### [HTTP response status codes for "Run an inference request attributed to an organization"](#run-an-inference-request-attributed-to-an-organization--status-codes)

Status code

Description

`200`

OK

### [Code samples for "Run an inference request attributed to an organization"](#run-an-inference-request-attributed-to-an-organization--code-samples)

#### Request example

post/orgs/{org}/inference/chat/completions

*   cURL
    

Copy to clipboard curl request example

`curl -L \ -X POST \ -H "Accept: application/vnd.github+json" \ -H "Authorization: Bearer <YOUR-TOKEN>" \ -H "X-GitHub-Api-Version: 2026-03-10" \ -H "Content-Type: application/json" \ https://models.github.ai/orgs/ORG/inference/chat/completions \ -d '{"model":"openai/gpt-4.1","messages":[{"role":"user","content":"What is the capital of France?"}]}'`

#### Response

*   Example response
    
*   Response schema
    

`Status: 200`

`{ "choices": [ { "message": { "content": "The capital of France is Paris.", "role": "assistant" } } ] }`

## [Run an inference request](#run-an-inference-request)

This endpoint allows you to run an inference request. The token used to authenticate must have the `models: read` permission if using a fine-grained PAT or GitHub App minted token. The request body should contain the model ID and the messages for the chat completion request. The response will include either a non-streaming or streaming response based on the request parameters.

### [Parameters for "Run an inference request"](#run-an-inference-request--parameters)

Headers

Name, Type, Description

`content-type` string Required

Setting to `application/json` is required.

`accept` string

Setting to `application/vnd.github+json` is recommended.

Query parameters

Name, Type, Description

`api-version` string

The API version to use. Optional, but required for some features.

Body parameters

Name, Type, Description

`model` string Required

ID of the specific model to use for the request. The model ID should be in the format of {publisher}/{model\_name} where "openai/gpt-4.1" is an example of a model ID. You can find supported models in the catalog/models endpoint.

`messages` array of objects Required

The collection of context messages associated with this chat completion request. Typical usage begins with a chat message for the System role that provides instructions for the behavior of the assistant, followed by alternating messages between the User and Assistant roles.

Properties of `messages`

Name, Type, Description

`role` string Required

The chat role associated with this message

Can be one of: `assistant`, `developer`, `system`, `user`

`content` string Required

The content of the message

`frequency_penalty` number

A value that influences the probability of generated tokens appearing based on their cumulative frequency in generated text. Positive values will make tokens less likely to appear as their frequency increases and decrease the likelihood of the model repeating the same statements verbatim. Supported range is \[-2, 2\].

`max_tokens` integer

The maximum number of tokens to generate in the completion. The token count of your prompt plus max\_tokens cannot exceed the model's context length. For example, if your prompt is 100 tokens and you set max\_tokens to 50, the API will return a completion with a maximum of 50 tokens.

`modalities` array of strings

The modalities that the model is allowed to use for the chat completions response. The default modality is text. Indicating an unsupported modality combination results in a 422 error. Supported values are: `text`, `audio`

`presence_penalty` number

A value that influences the probability of generated tokens appearing based on their existing presence in generated text. Positive values will make tokens less likely to appear when they already exist and increase the model's likelihood to output new tokens. Supported range is \[-2, 2\].

`response_format` object

The desired format for the response.

Can be one of these objects:

Name, Type, Description

`Object` object

Properties of `Object`

Name, Type, Description

`type` string

Can be one of: `text`, `json_object`

`Schema for structured JSON response` object

Properties of `Schema for structured JSON response`

Name, Type, Description

`type` string Required

The type of the response.

Value: `json_schema`

`json_schema` object Required

The JSON schema for the response.

`seed` integer

If specified, the system will make a best effort to sample deterministically such that repeated requests with the same seed and parameters should return the same result. Determinism is not guaranteed.

`stream` boolean

A value indicating whether chat completions should be streamed for this request.

Default: `false`

`stream_options` object

Whether to include usage information in the response. Requires stream to be set to true.

Properties of `stream_options`

Name, Type, Description

`include_usage` boolean

Whether to include usage information in the response.

Default: `false`

`stop` array of strings

A collection of textual sequences that will end completion generation.

`temperature` number

The sampling temperature to use that controls the apparent creativity of generated completions. Higher values will make output more random while lower values will make results more focused and deterministic. It is not recommended to modify temperature and top\_p for the same completion request as the interaction of these two settings is difficult to predict. Supported range is \[0, 1\]. Decimal values are supported.

`tool_choice` string

If specified, the model will configure which of the provided tools it can use for the chat completions response.

Can be one of: `auto`, `required`, `none`

`tools` array of objects

A list of tools the model may request to call. Currently, only functions are supported as a tool. The model may respond with a function call request and provide the input arguments in JSON format for that function.

Properties of `tools`

Name, Type, Description

`function` object

Properties of `function`

Name, Type, Description

`name` string

The name of the function to be called.

`description` string

A description of what the function does. The model will use this description when selecting the function and interpreting its parameters.

`parameters`

The parameters the function accepts, described as a JSON Schema object.

`type` string

Value: `function`

`top_p` number

An alternative to sampling with temperature called nucleus sampling. This value causes the model to consider the results of tokens with the provided probability mass. As an example, a value of 0.15 will cause only the tokens comprising the top 15% of probability mass to be considered. It is not recommended to modify temperature and top\_p for the same request as the interaction of these two settings is difficult to predict. Supported range is \[0, 1\]. Decimal values are supported.

### [HTTP response status codes for "Run an inference request"](#run-an-inference-request--status-codes)

Status code

Description

`200`

OK

### [Code samples for "Run an inference request"](#run-an-inference-request--code-samples)

#### Request example

post/inference/chat/completions

*   cURL
    

Copy to clipboard curl request example

`curl -L \ -X POST \ -H "Accept: application/vnd.github+json" \ -H "Authorization: Bearer <YOUR-TOKEN>" \ -H "X-GitHub-Api-Version: 2026-03-10" \ -H "Content-Type: application/json" \ https://models.github.ai/inference/chat/completions \ -d '{"model":"openai/gpt-4.1","messages":[{"role":"user","content":"What is the capital of France?"}]}'`

#### Response

*   Example response
    
*   Response schema
    

`Status: 200`

`{ "choices": [ { "message": { "content": "The capital of France is Paris.", "role": "assistant" } } ] }`

## Help and support

### Did you find what you needed?

 Yes No

[Privacy policy](/en/site-policy/privacy-policies/github-privacy-statement)

### Help us make these docs great!

All GitHub docs are open source. See something that's wrong or unclear? Submit a pull request.

[Make a contribution](https://github.com/github/docs/blob/main/content/rest/models/inference.md)

[Learn how to contribute](/contributing)

### Still need help?

[Ask the GitHub community](https://github.com/orgs/community/discussions)

[Contact support](https://support.github.com)

## Legal

*   © 2026 GitHub, Inc.
*   [Terms](/en/site-policy/github-terms/github-terms-of-service)
*   [Privacy](/en/site-policy/privacy-policies/github-privacy-statement)
*   [Status](https://www.githubstatus.com/)
*   [Pricing](https://github.com/pricing)
*   [Expert services](https://services.github.com)
*   [Blog](https://github.blog)