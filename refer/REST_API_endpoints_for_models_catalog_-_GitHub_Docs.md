# REST API endpoints for models catalog - GitHub Docs

> 来源: https://docs.github.com/en/rest/models/catalog?apiVersion=2026-03-10
> 摘要: Use the REST API to get a list of models available for use, including details like ID, supported input/output modalities, and rate limits.
> 站点: GitHub Docs

---

[Skip to main content](#main-content)

This article is also available in [Simplified Chinese](/zh/rest/models/catalog).

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
*   [Catalog](/en/rest/models/catalog "Catalog")

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
        *   [About GitHub Models catalog](#about-github-models-catalog)
        *   [List all models](#list-all-models)
    *   Embeddings
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
*   [Catalog](/en/rest/models/catalog "Catalog")

# REST API endpoints for models catalog

Use the REST API to get a list of models available for use, including details like ID, supported input/output modalities, and rate limits.

## [About GitHub Models catalog](#about-github-models-catalog)

You can use the REST API to explore available models in the GitHub Models catalog.

## [List all models](#list-all-models)

Get a list of models available for use, including details like supported input/output modalities, publisher, and rate limits.

### [HTTP response status codes for "List all models"](#list-all-models--status-codes)

Status code

Description

`200`

OK

### [Code samples for "List all models"](#list-all-models--code-samples)

#### Request example

get/catalog/models

*   cURL
    

Copy to clipboard curl request example

`curl -L \ -H "Accept: application/vnd.github+json" \ -H "Authorization: Bearer <YOUR-TOKEN>" \ -H "X-GitHub-Api-Version: 2026-03-10" \ https://models.github.ai/catalog/models`

#### Response

*   Example response
    
*   Response schema
    

`Status: 200`

`[ { "id": "openai/gpt-4.1", "name": "OpenAI GPT-4.1", "publisher": "OpenAI", "registry": "azure-openai", "summary": "gpt-4.1 outperforms gpt-4o across the board, with major gains in coding, instruction following, and long-context understanding", "html_url": "https://github.com/marketplace/models/azure-openai/gpt-4-1", "version": "2025-04-14", "capabilities": [ "streaming", "tool-calling" ], "limits": { "max_input_tokens": 1048576, "max_output_tokens": 32768 }, "rate_limit_tier": "high", "supported_input_modalities": [ "text", "image", "audio" ], "supported_output_modalities": [ "text" ], "tags": [ "multipurpose", "multilingual", "multimodal" ] } ]`

## Help and support

### Did you find what you needed?

 Yes No

[Privacy policy](/en/site-policy/privacy-policies/github-privacy-statement)

### Help us make these docs great!

All GitHub docs are open source. See something that's wrong or unclear? Submit a pull request.

[Make a contribution](https://github.com/github/docs/blob/main/content/rest/models/catalog.md)

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