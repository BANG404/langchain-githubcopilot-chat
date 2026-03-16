# REST API endpoints for model embeddings - GitHub Docs

> 来源: https://docs.github.com/en/rest/models/embeddings?apiVersion=2026-03-10
> 摘要: Use the REST API to work with embedding requests for models.
> 站点: GitHub Docs

---

[Skip to main content](#main-content)

This article is also available in [Simplified Chinese](/zh/rest/models/embeddings).

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
*   [Embeddings](/en/rest/models/embeddings "Embeddings")

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
        *   [Run an embedding request attributed to an organization](#run-an-embedding-request-attributed-to-an-organization)
        *   [Run an embedding request](#run-an-embedding-request)
    *   Inference
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
*   [Embeddings](/en/rest/models/embeddings "Embeddings")

# REST API endpoints for model embeddings

Use the REST API to work with embedding requests for models.

## [Run an embedding request attributed to an organization](#run-an-embedding-request-attributed-to-an-organization)

This endpoint allows you to run an embedding request attributed to a specific organization. You must be a member of the organization and have enabled models to use this endpoint. The token used to authenticate must have the `models: read` permission if using a fine-grained PAT or GitHub App minted token. The request body should contain the model ID and the input text(s) for the embedding request. The response will include the generated embeddings.

### [Parameters for "Run an embedding request attributed to an organization"](#run-an-embedding-request-attributed-to-an-organization--parameters)

Headers

Name, Type, Description

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

ID of the specific model to use for the request. The model ID should be in the format of {publisher}/{model\_name} where "openai/text-embedding-3-small" is an example of a model ID. You can find supported models in the catalog/models endpoint.

`input` string or array Required

Input text to embed, encoded as a string or array of strings. To embed multiple inputs in a single request, pass an array of strings. Each input must not exceed the max input tokens for the model, cannot be an empty string, and any array must be 2048 dimensions or less.

`encoding_format` string

The format to return the embeddings in. Can be either float or base64.

Default: `float`

Can be one of: `float`, `base64`

`dimensions` integer

The number of dimensions the resulting output embeddings should have. Only supported in text-embedding-3 and later models.

`user` string

A unique identifier representing your end-user, which can help us to monitor and detect abuse.

### [HTTP response status codes for "Run an embedding request attributed to an organization"](#run-an-embedding-request-attributed-to-an-organization--status-codes)

Status code

Description

`200`

OK

### [Code samples for "Run an embedding request attributed to an organization"](#run-an-embedding-request-attributed-to-an-organization--code-samples)

#### Request example

post/orgs/{org}/inference/embeddings

*   cURL
    

Copy to clipboard curl request example

`curl -L \ -X POST \ -H "Accept: application/vnd.github+json" \ -H "Authorization: Bearer <YOUR-TOKEN>" \ -H "X-GitHub-Api-Version: 2026-03-10" \ https://models.github.ai/orgs/ORG/inference/embeddings \ -d '{"model":"openai/text-embedding-3-small","input":["The food was delicious and the waiter was very friendly.","I had a great time at the restaurant."]}'`

#### Response

*   Example response
    
*   Response schema
    

`Status: 200`

`{ "object": "list", "data": [ { "object": "embedding", "index": 0, "embedding": [ 0.0023064255, -0.009327292, -0.0028842222 ] } ], "model": "openai/text-embedding-3-small", "usage": { "prompt_tokens": 8, "total_tokens": 8 } }`

## [Run an embedding request](#run-an-embedding-request)

This endpoint allows you to run an embedding request. The token used to authenticate must have the `models: read` permission if using a fine-grained PAT or GitHub App minted token. The request body should contain the model ID and the input text(s) for the embedding request. The response will include the generated embeddings.

### [Parameters for "Run an embedding request"](#run-an-embedding-request--parameters)

Headers

Name, Type, Description

`accept` string

Setting to `application/vnd.github+json` is recommended.

Query parameters

Name, Type, Description

`api-version` string

The API version to use. Optional, but required for some features.

Body parameters

Name, Type, Description

`model` string Required

ID of the specific model to use for the request. The model ID should be in the format of {publisher}/{model\_name} where "openai/text-embedding-3-small" is an example of a model ID. You can find supported models in the catalog/models endpoint.

`input` string or array Required

Input text to embed, encoded as a string or array of strings. To embed multiple inputs in a single request, pass an array of strings. Each input must not exceed the max input tokens for the model, cannot be an empty string, and any array must be 2048 dimensions or less.

`encoding_format` string

The format to return the embeddings in. Can be either float or base64.

Default: `float`

Can be one of: `float`, `base64`

`dimensions` integer

The number of dimensions the resulting output embeddings should have. Only supported in text-embedding-3 and later models.

`user` string

A unique identifier representing your end-user, which can help us to monitor and detect abuse.

### [HTTP response status codes for "Run an embedding request"](#run-an-embedding-request--status-codes)

Status code

Description

`200`

OK

### [Code samples for "Run an embedding request"](#run-an-embedding-request--code-samples)

#### Request example

post/inference/embeddings

*   cURL
    

Copy to clipboard curl request example

`curl -L \ -X POST \ -H "Accept: application/vnd.github+json" \ -H "Authorization: Bearer <YOUR-TOKEN>" \ -H "X-GitHub-Api-Version: 2026-03-10" \ https://models.github.ai/inference/embeddings \ -d '{"model":"openai/text-embedding-3-small","input":["The food was delicious and the waiter was very friendly.","I had a great time at the restaurant."]}'`

#### Response

*   Example response
    
*   Response schema
    

`Status: 200`

`{ "object": "list", "data": [ { "object": "embedding", "index": 0, "embedding": [ 0.0023064255, -0.009327292, -0.0028842222 ] } ], "model": "openai/text-embedding-3-small", "usage": { "prompt_tokens": 8, "total_tokens": 8 } }`

## Help and support

### Did you find what you needed?

 Yes No

[Privacy policy](/en/site-policy/privacy-policies/github-privacy-statement)

### Help us make these docs great!

All GitHub docs are open source. See something that's wrong or unclear? Submit a pull request.

[Make a contribution](https://github.com/github/docs/blob/main/content/rest/models/embeddings.md)

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