import * as path from "path";
import { Duration, RemovalPolicy, Stack, StackProps, CfnOutput } from "aws-cdk-lib";
import { Construct } from "constructs";
import * as lambda from "aws-cdk-lib/aws-lambda";
import * as ddb from "aws-cdk-lib/aws-dynamodb";
import * as iam from "aws-cdk-lib/aws-iam";
import * as s3 from "aws-cdk-lib/aws-s3";
import * as deploy from "aws-cdk-lib/aws-s3-deployment";
import * as cf from "aws-cdk-lib/aws-cloudfront";
import * as origins from "aws-cdk-lib/aws-cloudfront-origins";
import * as apigw from "aws-cdk-lib/aws-apigatewayv2";
import * as integrations from "aws-cdk-lib/aws-apigatewayv2-integrations";
import * as events from "aws-cdk-lib/aws-events";
import * as targets from "aws-cdk-lib/aws-events-targets";
import * as logs from "aws-cdk-lib/aws-logs";

export interface PlaygroundProps extends StackProps {
  deciderArn: string;
  originSecret: string;
  perIpHour: number;
  globalDay: number;
  keepWarm: boolean;
}

export class PlaygroundStack extends Stack {
  constructor(scope: Construct, id: string, props: PlaygroundProps) {
    super(scope, id, props);

    // rate-limit counters (per IP per hour, global per day) and running usage totals
    const table = new ddb.Table(this, "Usage", {
      partitionKey: { name: "pk", type: ddb.AttributeType.STRING },
      billingMode: ddb.BillingMode.PAY_PER_REQUEST,
      timeToLiveAttribute: "expires",
      removalPolicy: RemovalPolicy.DESTROY,
    });

    const fn = new lambda.Function(this, "Api", {
      runtime: lambda.Runtime.PYTHON_3_12,
      architecture: lambda.Architecture.ARM_64,
      handler: "app.handler",
      // ../backend/build holds app.py plus a current boto3 (see ../build.sh)
      code: lambda.Code.fromAsset(path.join(__dirname, "..", "..", "backend", "build")),
      timeout: Duration.seconds(29),
      memorySize: 512,
      logRetention: logs.RetentionDays.TWO_WEEKS,
      environment: {
        DECIDER_ARN: props.deciderArn,
        ROUTER_ARN: `arn:aws:bedrock:${this.region}:${this.account}:default-prompt-router/amazon.nova:1`,
        TABLE: table.tableName,
        PER_IP_HOUR: String(props.perIpHour),
        GLOBAL_DAY: String(props.globalDay),
        ORIGIN_SECRET: props.originSecret,
      },
    });
    table.grantReadWriteData(fn);
    fn.addToRolePolicy(new iam.PolicyStatement({
      actions: ["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"],
      resources: [
        "arn:aws:bedrock:*::foundation-model/amazon.nova-*",
        `arn:aws:bedrock:${this.region}:${this.account}:inference-profile/us.amazon.nova-*`,
        `arn:aws:bedrock:${this.region}:${this.account}:default-prompt-router/amazon.nova:1`,
      ],
    }));
    fn.addToRolePolicy(new iam.PolicyStatement({
      actions: ["bedrock:GetPromptRouter"],
      resources: [`arn:aws:bedrock:${this.region}:${this.account}:default-prompt-router/amazon.nova:1`],
    }));
    fn.addToRolePolicy(new iam.PolicyStatement({
      actions: ["bedrock-agentcore:InvokeAgentRuntime"],
      resources: [props.deciderArn, `${props.deciderArn}/*`],
    }));

    const api = new apigw.HttpApi(this, "Http", {
      defaultIntegration: new integrations.HttpLambdaIntegration("Fn", fn),
      createDefaultStage: false,
    });
    new apigw.HttpStage(this, "Stage", {
      httpApi: api, stageName: "$default", autoDeploy: true,
      throttle: { rateLimit: 5, burstLimit: 10 }, // account-wide backstop on top of the per-IP limits
    });

    if (props.keepWarm) {
      // keep one decider session warm: the decider takes about a minute to start a new microVM
      new events.Rule(this, "KeepWarm", {
        schedule: events.Schedule.rate(Duration.minutes(10)),
        targets: [new targets.LambdaFunction(fn, { event: events.RuleTargetInput.fromObject({ warm: true }) })],
      });
    }

    const site = new s3.Bucket(this, "Site", {
      blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
      enforceSSL: true,
      removalPolicy: RemovalPolicy.DESTROY,
      autoDeleteObjects: true,
    });
    const apiOriginPolicy = new cf.OriginRequestPolicy(this, "ApiOriginPolicy", {
      headerBehavior: cf.OriginRequestHeaderBehavior.allowList("CloudFront-Viewer-Address", "Content-Type"),
      queryStringBehavior: cf.OriginRequestQueryStringBehavior.none(),
      cookieBehavior: cf.OriginRequestCookieBehavior.none(),
    });
    const dist = new cf.Distribution(this, "Cdn", {
      defaultRootObject: "index.html",
      defaultBehavior: {
        origin: origins.S3BucketOrigin.withOriginAccessControl(site),
        viewerProtocolPolicy: cf.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
        responseHeadersPolicy: cf.ResponseHeadersPolicy.SECURITY_HEADERS,
      },
      additionalBehaviors: {
        "/api/*": {
          origin: new origins.HttpOrigin(`${api.apiId}.execute-api.${this.region}.amazonaws.com`, {
            customHeaders: { "x-origin-verify": props.originSecret },
          }),
          allowedMethods: cf.AllowedMethods.ALLOW_ALL,
          cachePolicy: cf.CachePolicy.CACHING_DISABLED,
          originRequestPolicy: apiOriginPolicy,
          viewerProtocolPolicy: cf.ViewerProtocolPolicy.HTTPS_ONLY,
        },
      },
      comment: "Strands Decider model-routing playground",
    });
    new deploy.BucketDeployment(this, "Web", {
      sources: [deploy.Source.asset(path.join(__dirname, "..", "..", "web"))],
      destinationBucket: site,
      distribution: dist,
      distributionPaths: ["/*"],
    });
    new CfnOutput(this, "Url", { value: `https://${dist.distributionDomainName}` });
    new CfnOutput(this, "ApiFunction", { value: fn.functionName });
  }
}
